"""
0616:no tcr
2025 06 26 VPP controller
VPP虚拟电厂优化系统

新增功能：投标比例管理
- 类似MATLAB中的硬编码参数，但更灵活
- 支持按设备类型或户号批量设置
- 每个设备都有默认的投标比例因子

使用示例:
1. 按设备类型设置: vpp.set_bid_ratios_by_type("DG", energy=0.9, reg_cap=0.2)
2. 按户号设置: vpp.set_bid_ratios_by_owner("USER123", reg_cap=0.15)  
3. 查看报告: vpp.print_bid_limits_report(owner_id="USER123")

这样在优化约束中就可以直接使用:
- device.get_bid_limit_energy_max()  # = bid_ratio_energy * Pout_max
- device.get_bid_limit_reg_cap_max() # = bid_ratio_reg_cap * Pout_max
"""

from pprint import pprint
from datetime import datetime, time

import numpy as np

from typing import List, Tuple, Any, Optional, Dict
import pandas as pd
from docplex.mp.model import Model
from docplex.mp.context import Context
from docplex.mp.conflict_refiner import ConflictRefiner
from flask import current_app

from .Equipment import PV, WIND, DG, ESS, DL, TCR, Equipment
import math
from .bidding_module import generate_multi_period_bidding_curves


class DeviceManager(object):
    def __init__(self):
        self.all_devices = {
            "PV": [],
            "WIND": [],
            "DG": [],
            "ESS": [],
            "DL": [],
            "TCR": [],
        }

    def create_pv_device(self, **kwargs) -> PV:
        pv = PV(**kwargs)
        self.all_devices["PV"].append(pv)
        return pv

    def create_wind_device(self, **kwargs) -> WIND:
        wind = WIND(**kwargs)
        self.all_devices["WIND"].append(wind)
        return wind

    def create_dg_device(self, **kwargs) -> DG:
        dg = DG(**kwargs)
        self.all_devices["DG"].append(dg)
        return dg

    def create_ess_device(self, **kwargs) -> ESS:
        ess = ESS(**kwargs)
        self.all_devices["ESS"].append(ess)
        return ess

    def create_dl_device(self, **kwargs) -> DL:
        dl = DL(**kwargs)
        self.all_devices["DL"].append(dl)
        return dl

    def create_tcr_device(self, **kwargs) -> TCR:
        tcr = TCR(**kwargs)
        self.all_devices["TCR"].append(tcr)
        return tcr

    def delete_device(self, device_id: str) -> bool:
        """
        根据设备ID删除设备,一次只能删除一个设备

        参数:
            device_id: 设备唯一标识符

        返回:
            bool: 是否成功删除
        """
        # 在所有设备类型中查找匹配的设备ID
        for device_type in self.all_devices:
            for i, device in enumerate(self.all_devices[device_type]):
                if device.DeviceID == device_id:
                    del self.all_devices[device_type][i]
                    return True

        return False

    def _get_devices_by_owner(self, owner_id: str = None) -> Dict[str, List[Equipment]]:
        """
        按户号获取设备信息,最佳实践输入单个户号,不建议省略
        参数:
            owner_id: 户号，如果为None则返回所有户号的设备
        返回:
            dict: 按户号组织的设备信息
        """
        if owner_id:
            # 返回指定户号的设备
            owner_devices = {
                "PV": [],
                "WIND": [],
                "DG": [],
                "ESS": [],
                "DL": [],
                "TCR": [],
            }
            for device_type, devices in self.all_devices.items():
                for device in devices:
                    if device.OwnerID == owner_id:
                        owner_devices[device_type].append(device)
            return owner_devices
        else:  # example:dict_keys(['USER123', 'FACTORY456']) if result.keys
            # 返回按户号分组的所有设备
            owners_devices = {}
            for device_type, devices in self.all_devices.items():
                for device in devices:
                    if device.OwnerID not in owners_devices:
                        owners_devices[device.OwnerID] = {
                            "PV": [],
                            "WIND": [],
                            "DG": [],
                            "ESS": [],
                            "DL": [],
                            "TCR": [],
                        }
                    owners_devices[device.OwnerID][device_type].append(device)
            return owners_devices

    def __calculate_aggregated_capacity(
        self, devices_dict: dict, only_active: bool = True
    ):
        """
        计算设备的聚合容量
        参数:
            devices_dict: 设备字典
            only_active: 是否只统计激活设备
        返回:
            dict: 聚合的容量信息
        """
        aggregated = {
            "total_devices": 0,
            "active_devices": 0,
            "total_capacity": {
                "Pout_min_sum": 0.0,
                "Pout_max_sum": 0.0,
                "Pin_min_sum": 0.0,
                "Pin_max_sum": 0.0,
            },
            "device_type_nums": {},  # 用于存储每种类型的数量
        }

        for device_type, devices in devices_dict.items():
            if not devices:
                continue

            # 初始化该设备类型的统计信息
            type_info = {
                "count": len(devices),
                "active_count": 0,
                "Pout_min_sum": 0.0,
                "Pout_max_sum": 0.0,
                "Pin_min_sum": 0.0,
                "Pin_max_sum": 0.0,
            }

            for device in devices:
                aggregated["total_devices"] += 1

                if device.Status == 1:
                    type_info["active_count"] += 1
                    aggregated["active_devices"] += 1

                if not only_active or device.Status == 1:
                    type_info["Pout_min_sum"] += device.Pout_min
                    type_info["Pout_max_sum"] += device.Pout_max
                    aggregated["total_capacity"]["Pout_min_sum"] += device.Pout_min
                    aggregated["total_capacity"]["Pout_max_sum"] += device.Pout_max

                    # 处理可输入功率的设备（如ESS, DL, TCR）
                    if hasattr(device, "Pin_min") and hasattr(device, "Pin_max"):
                        type_info["Pin_min_sum"] += getattr(device, "Pin_min", 0)
                        type_info["Pin_max_sum"] += getattr(device, "Pin_max", 0)
                        aggregated["total_capacity"]["Pin_min_sum"] += getattr(
                            device, "Pin_min", 0
                        )
                        aggregated["total_capacity"]["Pin_max_sum"] += getattr(
                            device, "Pin_max", 0
                        )

        return aggregated

    def _get_bid_limits_summary(self, owner_id: str = None, device_type: str = None):
        """
        获取投标限制摘要信息
        参数:
            owner_id: 户号 (可选)
            device_type: 设备类型 (可选)
        """
        summary = {}

        for dtype, devices in self.all_devices.items():
            if device_type and dtype != device_type:
                continue

            for device in devices:
                if owner_id and device.OwnerID != owner_id:
                    continue

                key = f"{device.OwnerID}-{device.DeviceType}"
                if key not in summary:
                    summary[key] = []

                summary[key].append(
                    {
                        "DeviceID": device.DeviceID,
                        "SimpleID": device.SimpleID,
                        "Status": device.Status,
                        "Pout_max": device.Pout_max,
                        "bid_limit_energy_max": device.get_bid_limit_energy_max(),
                        "bid_limit_reg_cap_max": device.get_bid_limit_reg_cap_max(),
                        "bid_limit_rev_max": device.get_bid_limit_rev_max(),
                        "bid_ratios": {
                            "energy": device.bid_ratio_energy,
                            "reg_cap": device.bid_ratio_reg_cap,
                            "rev": device.bid_ratio_rev,
                        },
                    }
                )

        return summary

    def get_owner_aggregated_capacity(
        self, owner_id: str = None, only_active: bool = True
    ) -> Dict[str, Dict]:
        """
        获取按户号得到的总装机量
        注意这是装机量，不是实际出力最大值
        参数:
            owner_id: 户号，如果为None则返回所有户号的聚合信息
            only_active: 是否只统计激活状态的设备
        返回:
            dict: 按户号和设备类型聚合的容量信息
        """
        result = {}

        if owner_id:
            # 单个户号的聚合信息
            owner_devices = self._get_devices_by_owner(owner_id)
            result[owner_id] = self.__calculate_aggregated_capacity(
                owner_devices, only_active
            )
        else:
            # 所有户号的聚合信息
            all_owners_devices = self._get_devices_by_owner()
            for owner, devices in all_owners_devices.items():
                result[owner] = self.__calculate_aggregated_capacity(
                    devices, only_active
                )

        return result

    # 设备级别-批量更新比例市场部分计算因子
    def update_bid_ratios_by_type(
        self, device_type: str, energy=None, reg_cap=None, rev=None
    ):
        """
        按设备类型批量更新投标比例因子
        参数:
            device_type: 设备类型 ("PV", "WIND", "DG", "ESS", "DL", "TCR")
            energy: 电能量投标比例因子
            reg_cap: 调频容量投标比例因子
            rev: 备用投标比例因子
        """
        if device_type not in self.all_devices:
            print(f"警告: 设备类型 {device_type} 不存在")
            return

        updated_count = 0
        for device in self.all_devices[device_type]:
            device.update_bid_ratios(energy=energy, reg_cap=reg_cap, rev=rev)
            updated_count += 1

        print(f"已更新 {device_type} 类型的 {updated_count} 台设备的投标比例因子")

    def update_bid_ratios_by_owner(
        self, owner_id: str, energy=None, reg_cap=None, rev=None
    ):
        """
        按户号批量更新投标比例因子
        参数:
            owner_id: 户号
            energy: 电能量投标比例因子
            reg_cap: 调频容量投标比例因子
            rev: 备用投标比例因子
        """
        updated_count = 0
        for device_type, devices in self.all_devices.items():
            for device in devices:
                if device.OwnerID == owner_id:
                    device.update_bid_ratios(energy=energy, reg_cap=reg_cap, rev=rev)
                    updated_count += 1

        print(f"已更新户号 {owner_id} 的 {updated_count} 台设备的投标比例因子")


class VirtualPowerPlantHost(object):
    time_dayahead_points = 96  # 96 points for dayahead
    time_intraday_points = 8  # 8 points for intraday (期望规划未来2小时)

    def __init__(self):
        self.vpp_devices = DeviceManager()  # 设备管理器
        self.optimization_result = None  # 优化结果
        self.bidding_result = None  # 投标结果

    def get_capacity_information_by_owner_id(
        self, owner_id: str = None
    ) -> Dict[str, dict]:
        """
        获取户号设备摘要信息 - 内部方法
        参数:
            owner_id: 户号，如果为None则返回所有户号的摘要
        返回:
            dict: 户号设备摘要
        """
        return self.vpp_devices.get_owner_aggregated_capacity(owner_id)

    def print_owner_summary(self, owner_id: str = None):
        """
        打印户号设备摘要信息

        参数:
            owner_id: 户号，如果为None则打印所有户号的摘要
        """
        summary = self.get_capacity_information_by_owner_id(owner_id)

        if owner_id:
            if owner_id in summary:
                self.__print_single_owner_summary(owner_id, summary[owner_id])
            else:
                print(f"未找到户号 {owner_id} 的设备信息")
        else:
            for owner, data in summary.items():
                self.__print_single_owner_summary(owner, data)
                print("-" * 50)

    def __print_single_owner_summary(self, owner_id: str, data: dict):
        """打印单个户号的设备摘要"""
        print(f"\n户号: {owner_id}")
        print(
            f"总设备数: {data[ 'total_devices' ]}, 激活设备数: {data[ 'active_devices' ]}"
        )
        print(
            f"输出容量范围: [{data[ 'total_capacity' ][ 'Pout_min_sum' ]:.4f}, {data[ 'total_capacity' ][ 'Pout_max_sum' ]:.4f}] kW"
        )
        print(
            f"输入容量范围: [{data[ 'total_capacity' ][ 'Pin_min_sum' ]:.4f}, {data[ 'total_capacity' ][ 'Pin_max_sum' ]:.4f}] kW"
        )

    def get_all_owners(self):
        """获取所有户号列表"""
        print("finding")
        all_owners = set()
        for device_type, devices in self.vpp_devices.all_devices.items():
            for device in devices:
                all_owners.add(device.OwnerID)
        print(f"all_owners: {all_owners}")
        return list(all_owners)

    # 控制限制接口 - 外部注入比例因子
    def set_bid_ratios_by_type(
        self,
        device_type: str,
        energy: object = None,
        reg_cap: object = None,
        rev: object = None,
    ) -> None:
        """按设备类型设置投标比例因子 - 外部接口"""
        return self.vpp_devices.update_bid_ratios_by_type(
            device_type, energy, reg_cap, rev
        )

    def set_bid_ratios_by_owner(
        self,
        owner_id: str,
        energy: object = None,
        reg_cap: object = None,
        rev: object = None,
    ) -> None:
        """按户号设置投标比例因子 - 外部接口"""
        return self.vpp_devices.update_bid_ratios_by_owner(
            owner_id, energy, reg_cap, rev
        )

    # 信息报告
    def _get_bid_limits_report(self, owner_id: str = None, device_type: str = None):
        """获取投标限制报告 - 内部方法"""
        return self.vpp_devices._get_bid_limits_summary(owner_id, device_type)

    def print_bid_limits_report(
        self, owner_id: str = None, device_type: str = None
    ) -> None:
        """打印投标限制报告"""
        summary = self._get_bid_limits_report(owner_id, device_type)
        print("\n=== 投标限制报告 ===")
        for key, devices in summary.items():
            print(f"\n{key}:")
            for device in devices:
                status_str = "激活" if device["Status"] == 1 else "停用"
                print(f"  {device[ 'SimpleID' ]} ({status_str}):")
                print(f"    装机容量: {device[ 'Pout_max' ]:.2f} kW")
                print(
                    f"    电能量限制: {device[ 'bid_limit_energy_max' ]:.2f} kW (比例: {device[ 'bid_ratios' ][ 'energy' ]:.3f})"
                )
                print(
                    f"    调频容量限制: {device[ 'bid_limit_reg_cap_max' ]:.2f} kW (比例: {device[ 'bid_ratios' ][ 'reg_cap' ]:.3f})"
                )
                print(
                    f"    备用限制: {device[ 'bid_limit_rev_max' ]:.2f} kW (比例: {device[ 'bid_ratios' ][ 'rev' ]:.3f})"
                )

    # Below is for optimization
    def _simulate_market_and_load(self, total: int) -> Tuple[List[float], ...]:
        """
        Args:
            total:时间序列总长度，通常为 VirtualPowerPlantHost.time_dayahead_points（96）。
        Returns:
            energy：模拟的电能量价格（元/MWh）
            regcap：模拟的调频容量价格（元/MWh）
            reserve：模拟的备用市场价格（元/MWh）
            load：模拟的本地负荷（kW）
        """
        import numpy as np, math

        day_var = 1 + 0.02 * (np.random.rand() - 0.5) * 2

        def gen(base, peaks, low_mul, high_mul, rnd_scale):
            result = []
            for i in range(total):
                hour = i // 4
                if any(start <= hour < end for start, end in peaks):
                    mul = high_mul + math.sin(i * math.pi / total)
                else:
                    mul = low_mul + math.sin(i * math.pi / total)
                result.append(
                    max(
                        base
                        * day_var
                        * mul
                        * (1 + rnd_scale * (np.random.rand() - 0.5)),
                        0,
                    )
                )
            return result

        energy = gen(300, [(6, 10), (17, 22)], 0.6, 1.2, 0.05)
        regcap = gen(250, [(7, 11), (17, 21)], 1.0, 1.5, 0.10)
        reserve = gen(200, [(6, 10), (18, 22)], 1.0, 1.6, 0.08)
        load = gen(50, [(6, 9), (17, 21)], 0.5, 1.3, 0.05)
        return energy, regcap, reserve, load

    def get_necessary_data_by_owner(self, owner_id: str):
        """
        重构后的方法：仅从台账加载数据并返回一个字典。
        不再修改self属性，成为一个纯粹的数据加载工具。
        keep for fullfill in future  and do not achieve
        """
        market_data = {}
        time_points = VirtualPowerPlantHost.time_dayahead_points

        # 1. 读取市场价格
        prices = self._read_market_prices_from_ledger()
        if prices:
            market_data.update(prices)
        else:
            print("WARN: 无法从台账加载市场价格，将使用全零数组。")
            market_data["energy"] = [0] * time_points
            market_data["reg_cap"] = [0] * time_points
            market_data["reserve"] = [0] * time_points

        # 2. 读取户号负荷
        busload = self._read_busload_from_ledger(owner_id)
        if busload:
            market_data["busload"] = busload
        else:
            print(f"WARN: 无法从台账加载户号 {owner_id} 的负荷，将使用全零数组。")
            market_data["busload"] = [0] * time_points

        return market_data

    def optimization_for_single_owner_with_device_selection(
        self,
        owner_id: str,
        market_and_load_data: Dict[str, List[float]],
        time_points=None,
        mode="day_ahead",
        start_ts: int = 0,
        EnergyBidRatio=0.7,
        RegCapRatio=0.2,
        RevRatio=0.2,
        participate_energy=True,
        participate_regulation=True,
        participate_reserve=True,
        enable_pv=True,
        enable_wind=True,
        enable_dg=True,
        enable_ess=True,
        enable_dl=True,
        enable_tcr=False,
    ):
        """
        支持设备类型自由选择的VPP优化函数 - 与optimization_for_single_owner逻辑完全一致
        Parameters (Virtual Power Plant Level):
            owner_id: 户号标识符
            owner_id: Identifier for the owner.
            time_points: 优化时间点数量（时间段数）
            time_points: Number of optimization time points (number of time slots).


            start_ts (int):
                    The absolute starting index (0-95) on the 24-hour timeline for the **period to be planned.
                    It defines the beginning of the target execution window.

                    Usage Scenarios:
                    - Day-Ahead (mode='day_ahead'): `start_ts` must be 0. This plans the entire upcoming day starting from 00:00.
                    - Intraday (mode='intra_day'): `start_ts` corresponds to the beginning of the future execution block.
                    - 例如，在当前 12:00，为了制定 14:00-16:00 的运行计划，应设置 `start_ts` 为 56 (代表 14:00-14:00+timepoint相关窗口时限)。优化计算本身在 14:00 之前完成。
                    - Scenario Analysis: Can be set to any future time slot to analyze specific periods.

            time_points (int):谨慎设置,可以使用默认值
                The duration of the optimization window, in number of 15-minute slots.
                - For day-ahead, this is typically 96 (24 hours).
                - For intraday, this corresponds to the length of the execution block, e.g., 8 for a 2-hour block.

            EnergyBidRatio: 电能量市场投标比例因子 (0.6~0.8),涉及到可用电量范围与装机量
            EnergyBidRatio: Energy market bidding ratio factor (0.6~0.8), related to the available energy range and installed capacity.
            RegCapRatio: 调频容量市场比例因子 (0.1~0.2)
            RegCapRatio: Regulation capacity market ratio factor (0.1~0.2).
            RevRatio: 备用市场比例因子 (0.15~0.25)
            RevRatio: Reserve market ratio factor (0.15~0.25).

            # 市场参与控制参数
            participate_energy: 是否参与电能量市场 (默认True)
            participate_energy: Whether to participate in the energy market (default True).
            participate_regulation: 是否参与调频市场(容量+里程绑定) (默认True)
            participate_regulation: Whether to participate in the regulation market (capacity + mileage binding) (default True).
            participate_reserve: 是否参与备用市场 (默认True)
            participate_reserve: Whether to participate in the reserve market (default True).

            # 设备类型选择参数
            enable_pv: 是否启用PV设备参与优化 (默认True)
            enable_pv: Whether to enable PV devices to participate in optimization (default True).
            enable_wind: 是否启用WIND设备参与优化 (默认True)
            enable_wind: Whether to enable WIND devices to participate in optimization (default True).
            enable_dg: 是否启用DG设备参与优化 (默认True)
            enable_dg: Whether to enable DG devices to participate in optimization (default True).
            enable_ess: 是否启用ESS设备参与优化 (默认True)
            enable_ess: Whether to enable ESS devices to participate in optimization (default True).
            enable_dl: 是否启用DL设备参与优化 (默认True)
            enable_dl: Whether to enable DL devices to participate in optimization (default True).

        """
        print(f"正在以 start_ts={start_ts} 和 time_points={time_points} 进行优化...")
        if not (0 <= start_ts < VirtualPowerPlantHost.time_dayahead_points):
            raise ValueError(f"start_ts 必须在 0~95 之间，目前是 {start_ts}")

        # =================================================================
        #  PART 1: DATA PREPARATION & STATE INIT (IF/ELSE BLOCK) 日前和日内
        # =================================================================

        if mode == "day_ahead":
            time_points = VirtualPowerPlantHost.time_dayahead_points  # 日前默认96
            if start_ts != 0:
                print(
                    f"Error: In day-ahead optimization mode, start_ts must be 0. Automatically correcting."
                )
                start_ts = 0
        else:  # 日内模式
            if time_points is None:
                time_points = VirtualPowerPlantHost.time_intraday_points
            else:
                time_points = time_points # 日内模式下，time_points 由用户传入 
        print(f"--- 模式: {mode}, 逻辑起点: {start_ts}, 优化时长: {time_points} ---")

        # 数据准备阶段: 直接使用从API层传入的数据
        self.EnergyMarketPrice = market_and_load_data.get(
            "energy_price", [0] * time_points
        )
        self.RegCapMarketPrice = market_and_load_data.get(
            "reg_cap_price", [0] * time_points
        )
        self.ReserveMarketPrice = market_and_load_data.get(
            "reserve_price", [0] * time_points
        )
        self.BusLoad = market_and_load_data.get("busload", [0] * time_points)

        print(f"信息: 已从API层接收市场和负荷数据并设置到优化实例中。")
        print(f"调试: 使用的电能量市场价格 (前10个点): {self.EnergyMarketPrice[:10]}")
        print(f"调试: 使用的负荷数据 (前10个点): {self.BusLoad[:10]}")
        print(f"调试: 使用的调频容量市场价格 (前10个点): {self.RegCapMarketPrice[:10]}")
        print(f"调试: 使用的备用市场价格 (前10个点): {self.ReserveMarketPrice[:10]}")

        # =================================================================
        # PART 2: CORE OPTIMIZATION MODELING (SHARED CODE)
        # =================================================================

        # 获取户号设备容量信息-VPP级别装机量
        capacity = self.get_capacity_information_by_owner_id(owner_id)[owner_id][
            "total_capacity"
        ]
        Vpp_pout_min, Vpp_pout_max = capacity["Pout_min_sum"], capacity["Pout_max_sum"]
        Vpp_pin_min, Vpp_pin_max = capacity["Pin_min_sum"], capacity["Pin_max_sum"]

        # 对实际可用电量进行限制
        Vpp_Pout_L = EnergyBidRatio * Vpp_pout_max
        Vpp_Pin_L = Vpp_pin_max
        Vpp_Rev_L = RevRatio * Vpp_pout_max
        Vpp_RegCap_L = RegCapRatio * Vpp_pout_max

        # vpp优化参与参数设置
        self.vpp_available_Quantity_min = -Vpp_Pin_L  # vpp可用的最小电量
        self.vpp_available_Quantity_max = Vpp_Pout_L  # vpp可用的最大电量
        self.vpp_available_RegCap_min = -Vpp_RegCap_L  # vpp可用的最小调频容量
        self.vpp_available_RegCap_max = Vpp_RegCap_L  # vpp可用的最大调频容量
        self.vpp_available_Rev_min = 0  # vpp可用的最小备用容量
        self.vpp_available_Rev_max = Vpp_Rev_L  # vpp可用的最大备用容量

        # 获取设备
        self.Vpp_temp_devices = self.vpp_devices._get_devices_by_owner(owner_id)

        # 建模
        mdl = Model(name="VPP_optimization_with_device_selection")

        print(
            r"""
        +---------------------+
        |  设备选择状态       |
        +---------------------+
        | PV   | {}          |
        | WIND | {}          |
        | DG   | {}          |
        | ESS  | {}          |
        | DL   | {}          |
        +---------------------+
        """.format(
                enable_pv, enable_wind, enable_dg, enable_ess, enable_dl
            )
        )

        # =================================================================
        # PART 3: ACCURATE ATTENTION
        # =================================================================
        # PV设备优化变量创建
        if "PV" in self.Vpp_temp_devices and enable_pv:  # 增加enable_pv检查
            num_pv_devices = len(self.Vpp_temp_devices["PV"])
            Energy_optimization_variables_pv_list = []
            RegCap_optimization_variables_pv_list = []
            Rev_optimization_variables_pv_list = []
            control_orders_pv_list = []
            for pv_unit in self.Vpp_temp_devices["PV"]:
                if participate_energy:
                    Energy_optimization_variables_pv_list.append(
                        mdl.continuous_var_list(
                            keys=time_points,
                            name=f"Energy_PV_{pv_unit.SimpleID}_Pout",
                            lb=0,
                        )
                    )
                else:
                    Energy_optimization_variables_pv_list.append(
                        [
                            mdl.continuous_var(
                                name=f"Energy_PV_{pv_unit.SimpleID}_Pout_{ts}",
                                lb=0,
                                ub=0,
                            )
                            for ts in range(time_points)
                        ]
                    )

                if participate_regulation:
                    RegCap_optimization_variables_pv_list.append(
                        mdl.continuous_var_list(
                            keys=time_points,
                            name=f"RegCap_PV_{pv_unit.SimpleID}_Pout",
                            lb=0,
                        )
                    )
                else:
                    RegCap_optimization_variables_pv_list.append(
                        [
                            mdl.continuous_var(
                                name=f"RegCap_PV_{pv_unit.SimpleID}_Pout_{ts}",
                                lb=0,
                                ub=0,
                            )
                            for ts in range(time_points)
                        ]
                    )

                if participate_reserve:
                    Rev_optimization_variables_pv_list.append(
                        mdl.continuous_var_list(
                            keys=time_points,
                            name=f"Rev_PV_{pv_unit.SimpleID}_Pout",
                            lb=0,
                        )
                    )
                else:
                    Rev_optimization_variables_pv_list.append(
                        [
                            mdl.continuous_var(
                                name=f"Rev_PV_{pv_unit.SimpleID}_Pout_{ts}", lb=0, ub=0
                            )
                            for ts in range(time_points)
                        ]
                    )

        # WIND设备优化变量创建
        if "WIND" in self.Vpp_temp_devices and enable_wind:  # 增加enable_wind检查
            num_wind_devices = len(self.Vpp_temp_devices["WIND"])
            Energy_optimization_variables_wind_list = []
            RegCap_optimization_variables_wind_list = []
            Rev_optimization_variables_wind_list = []
            control_orders_wind_list = []
            for wind_unit in self.Vpp_temp_devices["WIND"]:
                if participate_energy:
                    Energy_optimization_variables_wind_list.append(
                        mdl.continuous_var_list(
                            keys=time_points,
                            name=f"Energy_WIND_{wind_unit.SimpleID}_Pout",
                            lb=0,
                        )
                    )
                else:
                    Energy_optimization_variables_wind_list.append(
                        [
                            mdl.continuous_var(
                                name=f"Energy_WIND_{wind_unit.SimpleID}_Pout_{ts}",
                                lb=0,
                                ub=0,
                            )
                            for ts in range(time_points)
                        ]
                    )

                if participate_regulation:
                    RegCap_optimization_variables_wind_list.append(
                        mdl.continuous_var_list(
                            keys=time_points,
                            name=f"RegCap_WIND_{wind_unit.SimpleID}_Pout",
                            lb=0,
                        )
                    )
                else:
                    RegCap_optimization_variables_wind_list.append(
                        [
                            mdl.continuous_var(
                                name=f"RegCap_WIND_{wind_unit.SimpleID}_Pout_{ts}",
                                lb=0,
                                ub=0,
                            )
                            for ts in range(time_points)
                        ]
                    )

                if participate_reserve:
                    Rev_optimization_variables_wind_list.append(
                        mdl.continuous_var_list(
                            keys=time_points,
                            name=f"Rev_WIND_{wind_unit.SimpleID}_Pout",
                            lb=0,
                        )
                    )
                else:
                    Rev_optimization_variables_wind_list.append(
                        [
                            mdl.continuous_var(
                                name=f"Rev_WIND_{wind_unit.SimpleID}_Pout_{ts}",
                                lb=0,
                                ub=0,
                            )
                            for ts in range(time_points)
                        ]
                    )

        # DG设备优化变量创建
        if "DG" in self.Vpp_temp_devices and enable_dg:  # 增加enable_dg检查
            num_dg_devices = len(self.Vpp_temp_devices["DG"])
            Energy_optimization_variables_dg_list = []
            RegCap_optimization_variables_dg_list = []
            Rev_optimization_variables_dg_list = []
            control_orders_dg_list = []
            for dg_unit in self.Vpp_temp_devices["DG"]:
                if participate_energy:
                    Energy_optimization_variables_dg_list.append(
                        mdl.continuous_var_list(
                            keys=time_points,
                            name=f"Energy_DG_{dg_unit.SimpleID}_Pout",
                            lb=0,
                        )
                    )
                else:
                    Energy_optimization_variables_dg_list.append(
                        [
                            mdl.continuous_var(
                                name=f"Energy_DG_{dg_unit.SimpleID}_Pout_{ts}",
                                lb=0,
                                ub=0,
                            )
                            for ts in range(time_points)
                        ]
                    )

                if participate_regulation:
                    RegCap_optimization_variables_dg_list.append(
                        mdl.continuous_var_list(
                            keys=time_points,
                            name=f"RegCap_DG_{dg_unit.SimpleID}_Pout",
                            lb=0,
                        )
                    )
                else:
                    RegCap_optimization_variables_dg_list.append(
                        [
                            mdl.continuous_var(
                                name=f"RegCap_DG_{dg_unit.SimpleID}_Pout_{ts}",
                                lb=0,
                                ub=0,
                            )
                            for ts in range(time_points)
                        ]
                    )

                if participate_reserve:
                    Rev_optimization_variables_dg_list.append(
                        mdl.continuous_var_list(
                            keys=time_points,
                            name=f"Rev_DG_{dg_unit.SimpleID}_Pout",
                            lb=0,
                        )
                    )
                else:
                    Rev_optimization_variables_dg_list.append(
                        [
                            mdl.continuous_var(
                                name=f"Rev_DG_{dg_unit.SimpleID}_Pout_{ts}", lb=0, ub=0
                            )
                            for ts in range(time_points)
                        ]
                    )

        # ESS设备优化变量创建
        if "ESS" in self.Vpp_temp_devices and enable_ess:  # 增加enable_ess检查
            num_ess_devices = len(self.Vpp_temp_devices["ESS"])
            Energy_optimization_variables_ess_charge_list = []
            Energy_optimization_variables_ess_discharge_list = []
            RegCap_optimization_variables_ess_list = []
            Rev_optimization_variables_ess_list = []
            (
                Energy_optimization_variables_soc_list,
                ess_charge_binary_list,
                ess_discharge_binary_list,
            ) = ([], [], [])
            control_orders_ess_list = []
            for ess_unit in self.Vpp_temp_devices["ESS"]:
                if participate_energy:
                    Energy_optimization_variables_ess_charge_list.append(
                        mdl.continuous_var_list(
                            keys=time_points,
                            name=f"Energy_ESS_{ess_unit.SimpleID}_Pin",
                            lb=0,
                        )
                    )
                    Energy_optimization_variables_ess_discharge_list.append(
                        mdl.continuous_var_list(
                            keys=time_points,
                            name=f"Energy_ESS_{ess_unit.SimpleID}_Pout",
                            lb=0,
                        )
                    )
                    ess_charge_binary_list.append(
                        mdl.binary_var_list(
                            keys=time_points,
                            name=f"ESS_{ess_unit.SimpleID}_charge_binary",
                        )
                    )
                    ess_discharge_binary_list.append(
                        mdl.binary_var_list(
                            keys=time_points,
                            name=f"ESS_{ess_unit.SimpleID}_discharge_binary",
                        )
                    )
                else:
                    # not recommended
                    Energy_optimization_variables_ess_charge_list.append(
                        [
                            mdl.continuous_var(
                                name=f"Energy_ESS_{ess_unit.SimpleID}_Pin_{ts}",
                                lb=0,
                                ub=0,
                            )
                            for ts in range(time_points)
                        ]
                    )
                    Energy_optimization_variables_ess_discharge_list.append(
                        [
                            mdl.continuous_var(
                                name=f"Energy_ESS_{ess_unit.SimpleID}_Pout_{ts}",
                                lb=0,
                                ub=0,
                            )
                            for ts in range(time_points)
                        ]
                    )
                    ess_charge_binary_list.append(
                        [
                            mdl.binary_var(
                                name=f"ESS_{ess_unit.SimpleID}_charge_binary_{ts}"
                            )
                            for ts in range(time_points)
                        ]
                    )
                    ess_discharge_binary_list.append(
                        [
                            mdl.binary_var(
                                name=f"ESS_{ess_unit.SimpleID}_discharge_binary_{ts}"
                            )
                            for ts in range(time_points)
                        ]
                    )

                Energy_optimization_variables_soc_list.append(
                    mdl.continuous_var_list(
                        keys=time_points,
                        name=f"SOC_ESS_{ess_unit.SimpleID}",
                        lb=0,
                        ub=1,
                    )
                )

                if participate_regulation:
                    RegCap_optimization_variables_ess_list.append(
                        mdl.continuous_var_list(
                            keys=time_points,
                            name=f"RegCap_ESS_{ess_unit.SimpleID}",
                            lb=0,
                        )
                    )
                else:
                    RegCap_optimization_variables_ess_list.append(
                        [
                            mdl.continuous_var(
                                name=f"RegCap_ESS_{ess_unit.SimpleID}_{ts}", lb=0, ub=0
                            )
                            for ts in range(time_points)
                        ]
                    )

                if participate_reserve:
                    Rev_optimization_variables_ess_list.append(
                        mdl.continuous_var_list(
                            keys=time_points, name=f"Rev_ESS_{ess_unit.SimpleID}", lb=0
                        )
                    )
                else:
                    Rev_optimization_variables_ess_list.append(
                        [
                            mdl.continuous_var(
                                name=f"Rev_ESS_{ess_unit.SimpleID}_{ts}", lb=0, ub=0
                            )
                            for ts in range(time_points)
                        ]
                    )

        # DL设备优化变量创建
        if "DL" in self.Vpp_temp_devices and enable_dl:  # 增加enable_dl检查

            # debug
            print("--- 检查DL设备关键参数 ---")
            for dl_device in self.Vpp_temp_devices["DL"]:
                print(f"设备ID: {dl_device.SimpleID}")
                print(f"  - RequiredDemand: {dl_device.RequiredDemand}")
                print(f"  - InitialCapacity: {dl_device.InitialCapacity}")
                print(f"  - Cap_max: {dl_device.Cap_max}")
                print(f"  - Pin_max: {dl_device.Pin_max}")
                print(f"  - StartTime: {dl_device.StartTime}")
                print(f"  - EndTime: {dl_device.EndTime}")
            print("--------------------------")
            # end
            num_dl_devices = len(self.Vpp_temp_devices["DL"])
            Capacity_optimization_variables_dl_list = []
            Energy_optimization_variables_dl_list = []
            RegCap_optimization_variables_dl_list = []
            Rev_optimization_variables_dl_list = []
            control_orders_dl_list = []
            for dl_unit in self.Vpp_temp_devices["DL"]:
                if participate_energy:
                    Capacity_optimization_variables_dl_list.append(
                        mdl.continuous_var_list(
                            keys=time_points,
                            name=f"Capacity_DL_{dl_unit.SimpleID}",
                            lb=dl_unit.Cap_min,
                            ub=dl_unit.Cap_max,
                        )
                    )
                    Energy_optimization_variables_dl_list.append(
                        mdl.continuous_var_list(
                            keys=time_points,
                            name=f"Energy_DL_{dl_unit.SimpleID}_Pin",
                            lb=0,
                            ub=dl_unit.Pin_max,
                        )
                    )
                else:
                    pass # async
                if participate_regulation:
                    RegCap_optimization_variables_dl_list.append(
                        mdl.continuous_var_list(
                            keys=time_points, name=f"RegCap_DL_{dl_unit.SimpleID}_Pin", lb=0
                        )
                    )
                else:
                    RegCap_optimization_variables_dl_list.append(
                        [
                            mdl.continuous_var(name=f"RegCap_DL_{dl_unit.SimpleID}_Pin_{ts}", lb=0, ub=0)
                            for ts in range(time_points)
                        ]
                    )
                if participate_reserve:
                    Rev_optimization_variables_dl_list.append(
                    mdl.continuous_var_list(
                            keys=time_points, name=f"Rev_DL_{dl_unit.SimpleID}_Pin", lb=0
                        )
                    )
                else:
                    Rev_optimization_variables_dl_list.append(
                        [
                            mdl.continuous_var(name=f"Rev_DL_{dl_unit.SimpleID}_Pin_{ts}", lb=0, ub=0)
                            for ts in range(time_points)
                        ]
                    )

        # VPP层面优化变量创建
        if participate_energy:
            Vpp_output_vars = mdl.continuous_var_list(
                keys=time_points,
                name="vpp_optimal_power_output",
                lb=-Vpp_Pin_L,
                ub=Vpp_Pout_L,
            )
        else:
            Vpp_output_vars = [
                mdl.continuous_var(name=f"vpp_optimal_power_output_{ts}", lb=0, ub=0)
                for ts in range(time_points)
            ]

        if participate_regulation:
            Vpp_RegCap_vars = mdl.continuous_var_list(
                keys=time_points, name="vpp_regulation_capacity"
            )
        else:
            Vpp_RegCap_vars = [
                mdl.continuous_var(name=f"vpp_regulation_capacity_{ts}", lb=0, ub=0)
                for ts in range(time_points)
            ]

        if participate_reserve:
            Vpp_Rev_vars = mdl.continuous_var_list(keys=time_points, name="vpp_reserve")
        else:
            Vpp_Rev_vars = [
                mdl.continuous_var(name=f"vpp_reserve_{ts}", lb=0, ub=0)
                for ts in range(time_points)
            ]

        # =================== 约束条件构建 ===================

        # 初始化变量列表（处理未启用设备的情况）
        if not ("PV" in self.Vpp_temp_devices and enable_pv):
            Energy_optimization_variables_pv_list = []
            RegCap_optimization_variables_pv_list = []
            Rev_optimization_variables_pv_list = []

        if not ("WIND" in self.Vpp_temp_devices and enable_wind):
            Energy_optimization_variables_wind_list = []
            RegCap_optimization_variables_wind_list = []
            Rev_optimization_variables_wind_list = []

        if not ("DG" in self.Vpp_temp_devices and enable_dg):
            Energy_optimization_variables_dg_list = []
            RegCap_optimization_variables_dg_list = []
            Rev_optimization_variables_dg_list = []

        if not ("ESS" in self.Vpp_temp_devices and enable_ess):
            Energy_optimization_variables_ess_charge_list = []
            Energy_optimization_variables_ess_discharge_list = []
            RegCap_optimization_variables_ess_list = []
            Rev_optimization_variables_ess_list = []
            Energy_optimization_variables_soc_list = []
            ess_charge_binary_list = []
            ess_discharge_binary_list = []

        if not ("DL" in self.Vpp_temp_devices and enable_dl):
            Energy_optimization_variables_dl_list = []
            RegCap_optimization_variables_dl_list = []
            Rev_optimization_variables_dl_list = []
            Capacity_optimization_variables_dl_list = []

        # 约束条件构建
        for ts in range(time_points):
            # 传入的数据已经是处理好的窗口数据，长度为任务的 time_points
            if participate_energy:
                mdl.add_constraint(
                    Vpp_output_vars[ts] <= Vpp_Pout_L
                )  # vpp输出功率最值,非装机量
                mdl.add_constraint(Vpp_output_vars[ts] >= -Vpp_Pin_L)  # vpp最大吸收功率

            if participate_regulation:
                mdl.add_constraint(
                    Vpp_RegCap_vars[ts] <= Vpp_RegCap_L
                )  # VPP调频容量上限
                mdl.add_constraint(
                    Vpp_RegCap_vars[ts] >= -Vpp_RegCap_L
                )  # VPP调频容量下限

            if participate_reserve:
                mdl.add_constraint(Vpp_Rev_vars[ts] <= Vpp_Rev_L)  # VPP备用上限
                mdl.add_constraint(Vpp_Rev_vars[ts] >= 0)  # VPP备用下限

            # 联合-VPP层面的市场耦合约束
            if participate_energy and participate_regulation and participate_reserve:
                mdl.add_constraint(
                    Vpp_output_vars[ts] + Vpp_RegCap_vars[ts] + Vpp_Rev_vars[ts]
                    <= Vpp_Pout_L
                )
                mdl.add_constraint(
                    Vpp_output_vars[ts] - Vpp_RegCap_vars[ts] >= -Vpp_Pin_L
                )
            elif participate_energy and participate_regulation:
                mdl.add_constraint(
                    Vpp_output_vars[ts] + Vpp_RegCap_vars[ts] <= Vpp_Pout_L
                )
                mdl.add_constraint(
                    Vpp_output_vars[ts] - Vpp_RegCap_vars[ts] >= -Vpp_Pin_L
                )
            elif participate_energy and participate_reserve:
                mdl.add_constraint(Vpp_output_vars[ts] + Vpp_Rev_vars[ts] <= Vpp_Pout_L)
                mdl.add_constraint(Vpp_output_vars[ts] >= -Vpp_Pin_L)

            # ========== DG设备约束 - 只有启用时才添加 ==========
            if Energy_optimization_variables_dg_list and enable_dg:
                for index, dg_content in enumerate(
                    Energy_optimization_variables_dg_list
                ):
                    dg_device = self.Vpp_temp_devices["DG"][index]

                    # 电能量约束
                    energy_limit = dg_device.get_bid_limit_energy_max()
                    mdl.add_constraint(
                        Energy_optimization_variables_dg_list[index][ts] <= energy_limit
                    )
                    mdl.add_constraint(
                        Energy_optimization_variables_dg_list[index][ts]
                        >= dg_device.Pout_min
                    )

                    # 调频容量约束
                    reg_cap_limit = dg_device.get_bid_limit_reg_cap_max()
                    mdl.add_constraint(
                        RegCap_optimization_variables_dg_list[index][ts]
                        <= reg_cap_limit
                    )
                    mdl.add_constraint(
                        RegCap_optimization_variables_dg_list[index][ts] >= 0
                    )

                    # 备用约束
                    rev_limit = dg_device.get_bid_limit_rev_max()
                    mdl.add_constraint(
                        Rev_optimization_variables_dg_list[index][ts] <= rev_limit
                    )
                    mdl.add_constraint(
                        Rev_optimization_variables_dg_list[index][ts] >= 0
                    )

                    # 有功、调频和备用之间的上限耦合
                    mdl.add_constraint(
                        Energy_optimization_variables_dg_list[index][ts]
                        + RegCap_optimization_variables_dg_list[index][ts]
                        + Rev_optimization_variables_dg_list[index][ts]
                        <= energy_limit
                    )
                    # 有功、调频和备用之间的下限耦合
                    mdl.add_constraint(
                        Energy_optimization_variables_dg_list[index][ts]
                        - RegCap_optimization_variables_dg_list[index][ts]
                        >= dg_device.Pout_min
                    )

            # ========== PV设备约束 - 只有启用时才添加 ==========
            if Energy_optimization_variables_pv_list and enable_pv:
                for index, pv_content in enumerate(
                    Energy_optimization_variables_pv_list
                ):
                    pv_device = self.Vpp_temp_devices["PV"][index]
                    # 电能量约束
                    forecast_power = (
                        pv_device.get_forecast_power()
                    )  # >> self.window_forecast,noin,see 注入
                    # 获取预测功率限制，如果预测数据不足或不存在则使用设备投标限制

                    energy_limit = (
                        forecast_power[ts]
                        if forecast_power
                        else pv_device.get_bid_limit_energy_max()
                    )

                    mdl.add_constraint(
                        Energy_optimization_variables_pv_list[index][ts] <= energy_limit
                    )
                    mdl.add_constraint(
                        Energy_optimization_variables_pv_list[index][ts] >= 0
                    )

                    # 调频容量约束
                    reg_cap_limit = pv_device.get_bid_limit_reg_cap_max()
                    mdl.add_constraint(
                        RegCap_optimization_variables_pv_list[index][ts]
                        <= reg_cap_limit
                    )
                    mdl.add_constraint(
                        RegCap_optimization_variables_pv_list[index][ts] >= 0
                    )

                    # 备用约束
                    rev_limit = pv_device.get_bid_limit_rev_max()
                    mdl.add_constraint(
                        Rev_optimization_variables_pv_list[index][ts] <= rev_limit
                    )
                    mdl.add_constraint(
                        Rev_optimization_variables_pv_list[index][ts] >= 0
                    )

                    # 有功、调频和备用之间的耦合
                    mdl.add_constraint(
                        Energy_optimization_variables_pv_list[index][ts]
                        + RegCap_optimization_variables_pv_list[index][ts]
                        + Rev_optimization_variables_pv_list[index][ts]
                        <= energy_limit
                    )

            # ========== WIND设备约束 - 只有启用时才添加 ==========
            if Energy_optimization_variables_wind_list and enable_wind:
                for index, wind_content in enumerate(
                    Energy_optimization_variables_wind_list
                ):
                    wind_device = self.Vpp_temp_devices["WIND"][index]

                    # 电能量约束
                    forecast_power = wind_device.get_forecast_power()
                    energy_limit = (
                        forecast_power[ts]
                        if forecast_power
                        else wind_device.get_bid_limit_energy_max()
                    )
                    mdl.add_constraint(
                        Energy_optimization_variables_wind_list[index][ts]
                        <= energy_limit
                    )
                    mdl.add_constraint(
                        Energy_optimization_variables_wind_list[index][ts] >= 0
                    )

                    # 调频容量约束
                    reg_cap_limit = wind_device.get_bid_limit_reg_cap_max()
                    mdl.add_constraint(
                        RegCap_optimization_variables_wind_list[index][ts]
                        <= reg_cap_limit
                    )
                    mdl.add_constraint(
                        RegCap_optimization_variables_wind_list[index][ts] >= 0
                    )

                    # 备用约束
                    rev_limit = wind_device.get_bid_limit_rev_max()
                    mdl.add_constraint(
                        Rev_optimization_variables_wind_list[index][ts] <= rev_limit
                    )
                    mdl.add_constraint(
                        Rev_optimization_variables_wind_list[index][ts] >= 0
                    )

                    # 有功、调频和备用之间的耦合
                    mdl.add_constraint(
                        Energy_optimization_variables_wind_list[index][ts]
                        + RegCap_optimization_variables_wind_list[index][ts]
                        + Rev_optimization_variables_wind_list[index][ts]
                        <= energy_limit
                    )

            # ========== ESS设备约束 - 只有启用时才添加 ==========
            if Energy_optimization_variables_ess_charge_list and enable_ess:
                for index, ess_content in enumerate(
                    Energy_optimization_variables_ess_charge_list
                ):
                    ess_device = self.Vpp_temp_devices["ESS"][index]

                    # ESS充放电基本约束
                    mdl.add_constraint(
                        Energy_optimization_variables_ess_charge_list[index][ts]
                        <= ess_device.Pin_max
                    )
                    mdl.add_constraint(
                        Energy_optimization_variables_ess_discharge_list[index][ts]
                        <= ess_device.Pout_max
                    )
                    mdl.add_constraint(
                        Energy_optimization_variables_ess_charge_list[index][ts] >= 0
                    )
                    mdl.add_constraint(
                        Energy_optimization_variables_ess_discharge_list[index][ts] >= 0
                    )

                    # 充放电互斥约束（不能同时充放电）
                    mdl.add_constraint(
                        ess_charge_binary_list[index][ts]
                        + ess_discharge_binary_list[index][ts]
                        <= 1
                    )

                    # 充电功率与二进制变量的关系
                    mdl.add_constraint(
                        Energy_optimization_variables_ess_charge_list[index][ts]
                        <= ess_charge_binary_list[index][ts] * ess_device.Pin_max
                    )

                    # 放电功率与二进制变量的关系
                    mdl.add_constraint(
                        Energy_optimization_variables_ess_discharge_list[index][ts]
                        <= ess_discharge_binary_list[index][ts] * ess_device.Pout_max
                    )

                    # SOC约束（使用0-1的SOC比例）
                    mdl.add_constraint(
                        Energy_optimization_variables_soc_list[index][ts] <= 0.98
                    )
                    mdl.add_constraint(
                        Energy_optimization_variables_soc_list[index][ts] >= 0.05
                    )

                    # SOC动态约束（按照原函数的逻辑）
                    initial_soc = ess_device.InitialSOC
                    storage_capacity = ess_device.Capacity
                    transfer_efficiency = ess_device.TransferEfficiency
                    self_discharge_rate = ess_device.SelfDischargeRate
                    time_interval = 0.25  # 15分钟 = 0.25小时

                    if ts == 0:
                        if start_ts == 0:
                            start_soc_value = ess_device.InitialSOC
                        else:
                            start_soc_value = getattr(
                                ess_device, "CurrentSOC", ess_device.InitialSOC
                            )
                        mdl.add_constraint(
                            Energy_optimization_variables_soc_list[index][ts]
                            * storage_capacity
                            == start_soc_value
                            * storage_capacity
                            * self_discharge_rate  # 从初始状态也考虑自放电，更严谨
                            + Energy_optimization_variables_ess_charge_list[index][ts]
                            * transfer_efficiency
                            * time_interval
                            - Energy_optimization_variables_ess_discharge_list[index][
                                ts
                            ]
                            * time_interval
                            / transfer_efficiency
                        )
                    else:
                        mdl.add_constraint(
                            storage_capacity
                            * Energy_optimization_variables_soc_list[index][ts]
                            == storage_capacity
                            * Energy_optimization_variables_soc_list[index][ts - 1]
                            * self_discharge_rate
                            + Energy_optimization_variables_ess_charge_list[index][ts]
                            * transfer_efficiency
                            * time_interval
                            - Energy_optimization_variables_ess_discharge_list[index][
                                ts
                            ]
                            * time_interval
                            / transfer_efficiency,
                        )

                    # 调频容量约束
                    reg_cap_limit = ess_device.get_bid_limit_reg_cap_max()
                    mdl.add_constraint(
                        RegCap_optimization_variables_ess_list[index][ts]
                        <= reg_cap_limit
                    )
                    mdl.add_constraint(
                        RegCap_optimization_variables_ess_list[index][ts] >= 0
                    )

                    # 备用约束
                    rev_limit = ess_device.get_bid_limit_rev_max()
                    mdl.add_constraint(
                        Rev_optimization_variables_ess_list[index][ts] <= rev_limit
                    )
                    mdl.add_constraint(
                        Rev_optimization_variables_ess_list[index][ts] >= 0
                    )

                    # ESS与调频/备用市场的耦合约束
                    mdl.add_constraint(
                        Energy_optimization_variables_ess_discharge_list[index][ts]
                        + RegCap_optimization_variables_ess_list[index][ts]
                        + Rev_optimization_variables_ess_list[index][ts]
                        <= ess_device.Pout_max
                    )

            # ========== DL设备约束 - 只有启用时才添加 ==========
            # HSview
            if Energy_optimization_variables_dl_list and enable_dl:
                time_interval = 0.25  # 15分钟 = 0.25小时
                for index, dl_content in enumerate(
                    Energy_optimization_variables_dl_list
                ):
                    dl_device = self.Vpp_temp_devices["DL"][index]

                    # DL基本功率约束
                    mdl.add_constraint(
                        Energy_optimization_variables_dl_list[index][ts]
                        <= dl_device.Pin_max
                    )
                    mdl.add_constraint(
                        Energy_optimization_variables_dl_list[index][ts] >= 0
                    )

                    try:
                        work_start = dl_device.StartTime  # time对象 datetime.strptime(StartTime, "%H:%M").time()  >> HH:MM  ex 06:00
                        work_end = dl_device.EndTime  # time对象 datetime.strptime(EndTime, "%H:%M").time()  >> HH:MM  ex 15:30
                        print(f"work_start: {work_start}, work_end: {work_end}")
                        print(f"dl_device.StartTime: {dl_device.StartTime}, dl_device.EndTime: {dl_device.EndTime}")
                        print(f"dl_device.InitialCapacity: {dl_device.InitialCapacity}, dl_device.Cap_max: {dl_device.Cap_max}")
                        
                    except Exception as e:
                        work_start, work_end = time(0, 0), time(23, 45)

                    # 时间窗口约束 举证+simple算例分析
                    """
                    use for 0-95 (0:00-23:45)
                    suppose 
                        tp = 8 
                        st = 94 (23:30)
                        
                        1. ts = 0, global_ts = 94 
                            current_hour = 94//4 = 23 % 24 = 23
                            current_minute = (94 % 4)*15 = 30
                        2. ts = 1, global_ts = 95
                            current_hour = 95//4 = 23 % 24 = 23
                            current_minute = (95 % 4)*15 = 45
                        
                        rolling day
                        
                        3. ts = 2, global_ts = 96
                            current_hour = 96//4 = 24 % 24 = 0
                            current_minute = (96 % 4)*15 = 0
                        4. ts = 3, global_ts = 97
                            current_hour = 97//4 = 24 % 24 = 0
                            current_minute = (97 % 4)*15 = 15
                        5. ts = 4, global_ts = 98
                            current_hour = 98//4 = 24 % 24 = 0
                            current_minute = (98 % 4)*15 = 30

                            
                    """
                    # suppose ts=4,st=94 so now is 98 which is 00:30
                    global_ts = start_ts + ts # 94 + 4 = 98
                    current_hour = (global_ts // 4) % 24 # 98//4 = 24 % 24 = 0
                    current_minute = (global_ts % 4) * 15 # (98 % 4)*15 = 30
                    current_time_minutes = current_hour * 60 + current_minute # 0*60 + 30 = 30
                    # suppose work_start = 06:00, work_end = 15:30
                    work_start_minutes = work_start.hour * 60 + work_start.minute # 15:30 = 15*60 + 30 = 930
                    work_end_minutes = work_end.hour * 60 + work_end.minute # 5:00 = 5*60 + 0 = 300
                    is_work_time = False
                

                    # 处理跨天的情况
                    if work_end_minutes <= work_start_minutes: # may cross day 300<930 suupose for case2

                        is_work_time = (current_time_minutes >= work_start_minutes) or (
                            current_time_minutes < work_end_minutes
                        )
                    else:
                        # 正常工作时间
                        is_work_time = (
                            work_start_minutes
                            <= current_time_minutes
                            < work_end_minutes
                        )

                    if is_work_time:
                        # 工作时间内需要满足最小功率需求

                        mdl.add_constraint(
                            Energy_optimization_variables_dl_list[index][ts]
                            >= dl_device.Pin_min
                        )
                    else:
                        # 非工作时间功率为0
                        mdl.add_constraint(
                            Energy_optimization_variables_dl_list[index][ts] == 0
                        )

                    # --- 约束2: 核心修改 - 区分模式的容量动态约束 ---
                    if ts == 0:
                        if start_ts == 0:
                            start_capacity_value = dl_device.InitialCapacity*dl_device.Cap_max
                        else:
                            start_capacity_value = getattr(
                                dl_device,
                                "cumulative_capacity",
                                dl_device.InitialCapacity*dl_device.Cap_max,
                            )

                        mdl.add_constraint(  # 容量动态约束
                            Capacity_optimization_variables_dl_list[index][ts]
                            == start_capacity_value
                            + Energy_optimization_variables_dl_list[index][ts]
                            * time_interval,
                            ctname=f"c_dl_cap_initial_{dl_device.SimpleID}",
                        )
                    else:
                        # 对于 ts > 0, 累计逻辑是相同的
                        mdl.add_constraint(
                            Capacity_optimization_variables_dl_list[index][ts]
                            == Capacity_optimization_variables_dl_list[index][ts - 1]
                            + Energy_optimization_variables_dl_list[index][ts]
                            * time_interval,
                            ctname=f"c_dl_cap_dynamic_{dl_device.SimpleID}_{ts}",
                        )

                    # --- 约束3: Unchanged---
                    # 容量边界约束
                    mdl.add_constraint(
                        Capacity_optimization_variables_dl_list[index][ts]
                        <= dl_device.Cap_max
                    )
                    mdl.add_constraint(
                        Capacity_optimization_variables_dl_list[index][ts]
                        >= dl_device.Cap_min
                    )

                    # 调频容量约束
                    reg_cap_limit = dl_device.get_bid_limit_reg_cap_max()
                    mdl.add_constraint(
                        RegCap_optimization_variables_dl_list[index][ts]
                        <= reg_cap_limit
                    )
                    mdl.add_constraint(
                        RegCap_optimization_variables_dl_list[index][ts] >= 0
                    )

                    # 备用约束
                    rev_limit = dl_device.get_bid_limit_rev_max()
                    mdl.add_constraint(
                        Rev_optimization_variables_dl_list[index][ts] <= rev_limit
                    )
                    mdl.add_constraint(
                        Rev_optimization_variables_dl_list[index][ts] >= 0
                    )

                    # DL与调频/备用市场的耦合约束
                    mdl.add_constraint(
                        Energy_optimization_variables_dl_list[index][ts]
                        + RegCap_optimization_variables_dl_list[index][ts]
                        + Rev_optimization_variables_dl_list[index][ts]
                        <= dl_device.Pin_max
                    )
                    

        # =================== VPP聚合约束 ===================
        # 构建VPP能量平衡约束（根据设备选择动态构建）
        for ts in range(time_points):
            # 初始化能量平衡等式的组成部分
            generation_sum = 0  # 发电设备总输出
            load_sum = 0  # 负荷设备总输入

            # DG设备发电
            if Energy_optimization_variables_dg_list and enable_dg:
                for index in range(len(Energy_optimization_variables_dg_list)):
                    generation_sum += Energy_optimization_variables_dg_list[index][ts]

            # PV设备发电
            if Energy_optimization_variables_pv_list and enable_pv:
                for index in range(len(Energy_optimization_variables_pv_list)):
                    generation_sum += Energy_optimization_variables_pv_list[index][ts]

            # WIND设备发电
            if Energy_optimization_variables_wind_list and enable_wind:
                for index in range(len(Energy_optimization_variables_wind_list)):
                    generation_sum += Energy_optimization_variables_wind_list[index][ts]

            # ESS设备净输出（放电-充电）
            if Energy_optimization_variables_ess_charge_list and enable_ess:
                for index in range(len(Energy_optimization_variables_ess_charge_list)):
                    generation_sum += Energy_optimization_variables_ess_discharge_list[
                        index
                    ][ts]
                    load_sum += Energy_optimization_variables_ess_charge_list[index][ts]

            # DL设备负荷
            if Energy_optimization_variables_dl_list and enable_dl:
                for index in range(len(Energy_optimization_variables_dl_list)):
                    load_sum += Energy_optimization_variables_dl_list[index][ts]

            # VPP能量平衡约束：VPP输出 = 发电设备总输出 - 负荷设备总输入 - 本地负荷
            # 【核心修正】直接使用相对索引 ts，因为 BusLoad 已经是窗口数据
            local_bus_load = self.BusLoad[ts]
            mdl.add_constraint(
                Vpp_output_vars[ts] == generation_sum - load_sum - local_bus_load
            )

            # 初始化各个市场的聚合变量
            vpp_reg_cap_sum = 0
            vpp_rev_sum = 0

            # DG设备
            if "DG" in self.Vpp_temp_devices and enable_dg:
                for i in range(len(self.Vpp_temp_devices["DG"])):
                    if participate_regulation:
                        vpp_reg_cap_sum += RegCap_optimization_variables_dg_list[i][ts]
                    if participate_reserve:
                        vpp_rev_sum += Rev_optimization_variables_dg_list[i][ts]

            # PV设备
            if "PV" in self.Vpp_temp_devices and enable_pv:
                for i in range(len(self.Vpp_temp_devices["PV"])):
                    if participate_regulation:
                        vpp_reg_cap_sum += RegCap_optimization_variables_pv_list[i][ts]
                    if participate_reserve:
                        vpp_rev_sum += Rev_optimization_variables_pv_list[i][ts]

            # WIND设备
            if "WIND" in self.Vpp_temp_devices and enable_wind:
                for i in range(len(self.Vpp_temp_devices["WIND"])):
                    if participate_regulation:
                        vpp_reg_cap_sum += RegCap_optimization_variables_wind_list[i][
                            ts
                        ]
                    if participate_reserve:
                        vpp_rev_sum += Rev_optimization_variables_wind_list[i][ts]

            # ESS设备
            if "ESS" in self.Vpp_temp_devices and enable_ess:
                for i in range(len(self.Vpp_temp_devices["ESS"])):
                    if participate_regulation:
                        vpp_reg_cap_sum += RegCap_optimization_variables_ess_list[i][ts]
                    if participate_reserve:
                        vpp_rev_sum += Rev_optimization_variables_ess_list[i][ts]

            # DL设备
            if "DL" in self.Vpp_temp_devices and enable_dl:
                if ts == 0:
                    print(self.Vpp_temp_devices["DL"])
                for i in range(len(self.Vpp_temp_devices["DL"])):
                    if participate_regulation:
                        vpp_reg_cap_sum += RegCap_optimization_variables_dl_list[i][ts]
                    if participate_reserve:
                        vpp_rev_sum += Rev_optimization_variables_dl_list[i][ts]

            # 添加聚合约束
            if participate_regulation:
                mdl.add_constraint(
                    Vpp_RegCap_vars[ts] == vpp_reg_cap_sum,
                    ctname=f"c_vpp_reg_cap_balance_{ts}",
                )
            if participate_reserve:
                mdl.add_constraint(
                    Vpp_Rev_vars[ts] == vpp_rev_sum, ctname=f"c_vpp_rev_balance_{ts}"
                )

        # =================== 目标函数构建 ===================
        print("开始构建目标函数...")
        # --- 1. 计算与大网交易的净收入/成本 (核心) 换算为MW尺度---
        grid_transaction_revenue = mdl.sum([])  #
        time_interval = 0.25  # 15分钟 = 0.25小时
        if participate_energy:
            energy_revenue_terms = []
            for ts in range(time_points):
                price_per_mwh = self.EnergyMarketPrice[ts]
                energy_revenue_terms.append(
                    Vpp_output_vars[ts] * price_per_mwh * 0.00025
                )
            grid_transaction_revenue += mdl.sum(energy_revenue_terms)

        if participate_regulation:
            regulation_revenue_terms = []
            for ts in range(time_points):
                reg_cap_price_per_mwh = self.RegCapMarketPrice[ts]
                regulation_revenue_terms.append(
                    Vpp_RegCap_vars[ts] * reg_cap_price_per_mwh * 0.001
                )
            grid_transaction_revenue += mdl.sum(regulation_revenue_terms)

        if participate_reserve:
            reserve_revenue_terms = []
            for ts in range(time_points):
                reserve_price_per_mwh = self.ReserveMarketPrice[ts]
                reserve_revenue_terms.append(
                    Vpp_Rev_vars[ts] * reserve_price_per_mwh * 0.00025
                )
            grid_transaction_revenue += mdl.sum(reserve_revenue_terms)

        # -------------计算VPP的内部运行成本 (与大网交易无关)设备这些运行成本应该以KW尺度录入------------
        internal_operating_cost = mdl.sum([])
        time_interval = 0.25  # 15分钟 = 0.25h

        # 1. DG(发电机)的成本
        if "DG" in self.Vpp_temp_devices and enable_dg:
            dg_cost_terms = []
            for index, dg_device in enumerate(self.Vpp_temp_devices["DG"]):
                for ts in range(time_points):
                    total_cost_at_ts = 0

                    # 电能量成本 (燃料)
                    P_energy_kw = Energy_optimization_variables_dg_list[index][ts]
                    hourly_cost = (
                        getattr(dg_device, "Cost_energy_a", 0)
                        + getattr(dg_device, "Cost_energy_b", 0) * P_energy_kw
                        + getattr(dg_device, "Cost_energy_c", 0) * P_energy_kw**2
                    )
                    total_cost_at_ts += hourly_cost * time_interval

                    # 辅助服务成本
                    if participate_regulation:
                        P_reg_cap_kw = RegCap_optimization_variables_dg_list[index][ts]

                        total_cost_at_ts += (
                            P_reg_cap_kw
                            * time_interval
                            * getattr(dg_device, "Cost_Reg_Cap", 0)
                        )

                    if participate_reserve:
                        P_rev_kw = Rev_optimization_variables_dg_list[index][ts]
                        total_cost_at_ts += (
                            P_rev_kw * time_interval * getattr(dg_device, "Cost_Rev", 0)
                        )

                    dg_cost_terms.append(total_cost_at_ts)
            internal_operating_cost += mdl.sum(dg_cost_terms)

        # 2. ESS(储能)的成本
        if "ESS" in self.Vpp_temp_devices and enable_ess:
            ess_cost_terms = []
            for index, ess_device in enumerate(self.Vpp_temp_devices["ESS"]):
                for ts in range(time_points):
                    total_cost_at_ts = 0

                    # 电能量成本 (充放电损耗)
                    charge_kw = Energy_optimization_variables_ess_charge_list[index][ts]
                    discharge_kw = Energy_optimization_variables_ess_discharge_list[
                        index
                    ][ts]
                    throughput_kwh = (charge_kw + discharge_kw) * time_interval
                    total_cost_at_ts += throughput_kwh * getattr(
                        ess_device, "Cost_energy", 0
                    )

                    # 辅助服务成本
                    if participate_regulation:
                        P_reg_cap_kw = RegCap_optimization_variables_ess_list[index][ts]

                        total_cost_at_ts += (
                            P_reg_cap_kw
                            * time_interval
                            * getattr(ess_device, "Cost_Reg_Cap", 0)
                        )

                    if participate_reserve:
                        P_rev_kw = Rev_optimization_variables_ess_list[index][ts]
                        total_cost_at_ts += (
                            P_rev_kw
                            * time_interval
                            * getattr(ess_device, "Cost_Rev", 0)
                        )

                    ess_cost_terms.append(total_cost_at_ts)
            internal_operating_cost += mdl.sum(ess_cost_terms)

        # 3. PV(光伏)的成本
        if "PV" in self.Vpp_temp_devices and enable_pv:
            pv_cost_terms = []
            for index, pv_device in enumerate(self.Vpp_temp_devices["PV"]):
                for ts in range(time_points):
                    total_cost_at_ts = 0

                    # 光伏的发电边际成本通常视为0
                    total_cost_at_ts += 0

                    # 但提供辅助服务可能有成本 (比如逆变器损耗增加)
                    if participate_regulation:
                        P_reg_cap_kw = RegCap_optimization_variables_pv_list[index][ts]
                        total_cost_at_ts += (
                            P_reg_cap_kw
                            * time_interval
                            * getattr(pv_device, "Cost_Reg_Cap", 0)
                        )
                    if participate_reserve:
                        P_rev_kw = Rev_optimization_variables_pv_list[index][ts]
                        total_cost_at_ts += (
                            P_rev_kw * time_interval * getattr(pv_device, "Cost_Rev", 0)
                        )

                    pv_cost_terms.append(total_cost_at_ts)
            internal_operating_cost += mdl.sum(pv_cost_terms)

        # 4. WIND(风电)的成本
        if "WIND" in self.Vpp_temp_devices and enable_wind:
            wind_cost_terms = []
            for index, wind_device in enumerate(self.Vpp_temp_devices["WIND"]):
                for ts in range(time_points):
                    total_cost_at_ts = 0

                    # 风电的发电边际成本通常视为0
                    total_cost_at_ts += 0

                    # 辅助服务成本
                    if participate_regulation:
                        P_reg_cap_kw = RegCap_optimization_variables_wind_list[index][
                            ts
                        ]

                        total_cost_at_ts += (
                            P_reg_cap_kw
                            * time_interval
                            * getattr(wind_device, "Cost_Reg_Cap", 0)
                        )

                    if participate_reserve:
                        P_rev_kw = Rev_optimization_variables_wind_list[index][ts]
                        total_cost_at_ts += (
                            P_rev_kw
                            * time_interval
                            * getattr(wind_device, "Cost_Rev", 0)
                        )

                    wind_cost_terms.append(total_cost_at_ts)
            internal_operating_cost += mdl.sum(wind_cost_terms)

        # 5. DL(可控负荷)的成本
        if "DL" in self.Vpp_temp_devices and enable_dl:
            dl_cost_terms = []
            for index, dl_device in enumerate(self.Vpp_temp_devices["DL"]):
                for ts in range(time_points):
                    total_cost_at_ts = 0

                    # DL本身是负荷，其用电成本已体现在电网交易中，内部成本通常是0
                    # 除非有特殊的"不舒适度"成本或中断成本
                    total_cost_at_ts += 0

                    # 辅助服务成本
                    if participate_regulation:
                        P_reg_cap_kw = RegCap_optimization_variables_dl_list[index][ts]
                        total_cost_at_ts += (
                            P_reg_cap_kw
                            * time_interval
                            * getattr(dl_device, "Cost_Reg_Cap", 0)
                        )

                    if participate_reserve:
                        P_rev_kw = Rev_optimization_variables_dl_list[index][ts]
                        total_cost_at_ts += (
                            P_rev_kw * time_interval * getattr(dl_device, "Cost_Rev", 0)
                        )

                    dl_cost_terms.append(total_cost_at_ts)
            internal_operating_cost += mdl.sum(dl_cost_terms)

        # 6. TCR(温控负荷)的成本 (占位)
        # 假设有TCR变量列表：Energy_optimization_variables_tcr_list 等
        if "TCR" in self.Vpp_temp_devices and enable_tcr:
            # 此处添加TCR的成本计算逻辑，结构与DL类似
            pass
        # HSview
        # --- 新增：为ESS添加期末SOC软约束（惩罚项）---
        ess_terminal_soc_penalty = mdl.sum([])
        if mode != "day_ahead" and enable_ess:
            SOC_PENALTY_FACTOR = 100
            TARGET_SOC = 0.5
            # --- 构建惩罚表达式 ---
            penalty_terms = []
            for index, ess_device in enumerate(self.Vpp_temp_devices["ESS"]):
                # 获取期末SOC的优化变量
                terminal_soc_var = Energy_optimization_variables_soc_list[index][
                    time_points - 1
                ]
                # 计算惩罚成本：因子 * (实际值 - 目标值)^2
                penalty = SOC_PENALTY_FACTOR * (terminal_soc_var - TARGET_SOC) ** 2
                penalty_terms.append(penalty)
            # 将所有ESS设备的惩罚项加总
            if penalty_terms:
                ess_terminal_soc_penalty = mdl.sum(penalty_terms)
        
        # ==================== 【为DL任务添加奖励】 ====================
        # --- 2. DL任务完成软约束（可行性缺口惩罚）---
        dl_feasibility_penalty = mdl.sum([ ])
        if "DL" in self.Vpp_temp_devices and enable_dl:
            # 定义惩罚系数 (元/kWh)，代表每缺少1kWh的未来任务保障，愿意付出的代价
            # 这个值应该有意义，比如设为当地平均电价的2-3倍
            FEASIBILITY_PENALTY_FACTOR_PER_KWH = 1.0  # 例如，1元/kWh

            penalty_terms = [ ]
            for index, dl_device in enumerate(self.Vpp_temp_devices.get("DL", [ ])):

                # --- lookahead约束的计算逻辑，现在移到这里 ---
                required_demand = dl_device.RequiredDemand
                work_end_time = getattr(dl_device, "EndTime", time(23, 45)) # case1 >suppose is 22:15
                work_start_time = getattr(dl_device, "StartTime", time(0, 0)) # case1> suppose is 15:30
                max_charge_power = dl_device.Pin_max
                time_interval = 0.25
                # suppose st=70 so now is 70 
                deadline_global_ts = (work_end_time.hour * 60 + work_end_time.minute) // 15 # case1> 22*60 + 15 = 1335 //15 = 89
                first_future_ts = start_ts + time_points # 
                future_workable_slots = 0

                if first_future_ts <= deadline_global_ts:
                    work_start_minutes = work_start_time.hour * 60 + work_start_time.minute # 15*60 + 30 = 930
                    work_end_minutes = work_end_time.hour * 60 + work_end_time.minute # 22*60 + 15 = 1335
                    for future_ts_check in range(first_future_ts, deadline_global_ts + 1):
                        hour = (future_ts_check // 4) % 24
                        minute = (future_ts_check % 4) * 15
                        current_time_minutes = hour * 60 + minute
                        is_work_time = False
                        if work_end_minutes > work_start_minutes:
                            is_work_time = (work_start_minutes <= current_time_minutes < work_end_minutes)
                        else:
                            is_work_time = (current_time_minutes >= work_start_minutes) or (
                                        current_time_minutes < work_end_minutes)
                        if is_work_time:
                            future_workable_slots += 1

                future_potential_capacity = future_workable_slots * max_charge_power * time_interval
                # --- 计算逻辑结束 ---

                # 定义一个偏差变量，代表"任务缺口"，单位是kWh
                shortfall_var = mdl.continuous_var(name=f"dl_shortfall_{dl_device.SimpleID}", lb=0)

                # 建立关系：期末容量 + 未来潜力 + 任务缺口 >= 总需求
                mdl.add_constraint(
                    Capacity_optimization_variables_dl_list[ index ][ time_points - 1 ]
                    + future_potential_capacity
                    + shortfall_var
                    >= required_demand,
                    ctname=f"c_dl_soft_feasibility_{dl_device.SimpleID}"
                )

                # 将缺口变量乘以惩罚系数，加入到惩罚项中
                penalty_terms.append(shortfall_var * FEASIBILITY_PENALTY_FACTOR_PER_KWH)

            if penalty_terms:
                dl_feasibility_penalty = mdl.sum(penalty_terms)
       
        
        
        # 总利润 = 电网交易净收入 - 内部运行成本
        mdl.maximize(
            grid_transaction_revenue
            - internal_operating_cost
            - ess_terminal_soc_penalty
            - dl_feasibility_penalty
            
        )

        # =================== 模型求解 ===================
        try:
            # 设置求解器参数
            mdl.parameters.timelimit = 1000

            # 执行求解
            solution = mdl.solve()
            print(f"电网交易收益: {grid_transaction_revenue.solution_value}")
            print(f"内部运行成本: {internal_operating_cost.solution_value}")
            print(f"ESS期末惩罚: {ess_terminal_soc_penalty.solution_value}")

            if solution:

                # 构建结构化返回结果
                result = {
                    "status": "optimal",
                    "summary": {
                        "mode": mode,
                        "start_ts": start_ts,  # 新增：记录起始时间点
                        "objective_value": solution.get_objective_value(),
                        "internal_cost": internal_operating_cost.solution_value,
                        "external_revenue": grid_transaction_revenue.solution_value,
                        "net_profit": grid_transaction_revenue.solution_value
                        - internal_operating_cost.solution_value,
                        "owner_id": owner_id,
                        "time_points": time_points,
                        "market_participation": {
                            "energy": participate_energy,
                            "regulation": participate_regulation,
                            "reserve": participate_reserve,
                        },
                        "device_selection": {
                            "pv": enable_pv and "PV" in self.Vpp_temp_devices,
                            "wind": enable_wind and "WIND" in self.Vpp_temp_devices,
                            "dg": enable_dg and "DG" in self.Vpp_temp_devices,
                            "ess": enable_ess and "ESS" in self.Vpp_temp_devices,
                            "dl": enable_dl and "DL" in self.Vpp_temp_devices,
                            # "tcr": enable_tcr and "TCR" in self.Vpp_temp_devices
                        },
                    },
                    "vpp_data": {
                        "energy_output": [
                            Vpp_output_vars[ts].solution_value
                            for ts in range(time_points)
                        ],
                        "regulation_capacity": (
                            [
                                Vpp_RegCap_vars[ts].solution_value
                                for ts in range(time_points)
                            ]
                            if participate_regulation
                            else []
                        ),
                        "reserve": (
                            [
                                Vpp_Rev_vars[ts].solution_value
                                for ts in range(time_points)
                            ]
                            if participate_reserve
                            else []
                        ),
                    },
                    "device_data": {},
                }

                # 添加各设备类型的详细数据
                # DG设备数据
                if Energy_optimization_variables_dg_list and enable_dg:
                    result["device_data"]["dg"] = {}
                    for index, dg_device in enumerate(self.Vpp_temp_devices["DG"]):
                        device_id = dg_device.SimpleID
                        result["device_data"]["dg"][device_id] = {
                            "device_info": {
                                "DeviceID": dg_device.DeviceID,
                                "SimpleID": dg_device.SimpleID,
                                "Status": dg_device.Status,
                                "Pout_max": dg_device.Pout_max,
                                "Pout_min": dg_device.Pout_min,
                            },
                            "energy_output": [
                                Energy_optimization_variables_dg_list[index][
                                    ts
                                ].solution_value
                                for ts in range(time_points)
                            ],
                            "regulation_capacity": (
                                [
                                    RegCap_optimization_variables_dg_list[index][
                                        ts
                                    ].solution_value
                                    for ts in range(time_points)
                                ]
                                if participate_regulation
                                else []
                            ),
                            "reserve": (
                                [
                                    Rev_optimization_variables_dg_list[index][
                                        ts
                                    ].solution_value
                                    for ts in range(time_points)
                                ]
                                if participate_reserve
                                else []
                            ),
                        }

                # PV设备数据
                if Energy_optimization_variables_pv_list and enable_pv:
                    result["device_data"]["pv"] = {}
                    for index, pv_device in enumerate(self.Vpp_temp_devices["PV"]):
                        device_id = pv_device.SimpleID
                        result["device_data"]["pv"][device_id] = {
                            "device_info": {
                                "DeviceID": pv_device.DeviceID,
                                "SimpleID": pv_device.SimpleID,
                                "Status": pv_device.Status,
                                "Pout_max": pv_device.Pout_max,
                            },
                            "energy_output": [
                                Energy_optimization_variables_pv_list[index][
                                    ts
                                ].solution_value
                                for ts in range(time_points)
                            ],
                            "regulation_capacity": (
                                [
                                    RegCap_optimization_variables_pv_list[index][
                                        ts
                                    ].solution_value
                                    for ts in range(time_points)
                                ]
                                if participate_regulation
                                else []
                            ),
                            "reserve": (
                                [
                                    Rev_optimization_variables_pv_list[index][
                                        ts
                                    ].solution_value
                                    for ts in range(time_points)
                                ]
                                if participate_reserve
                                else []
                            ),
                        }

                # WIND设备数据
                if Energy_optimization_variables_wind_list and enable_wind:
                    result["device_data"]["wind"] = {}
                    for index, wind_device in enumerate(self.Vpp_temp_devices["WIND"]):
                        device_id = wind_device.SimpleID
                        result["device_data"]["wind"][device_id] = {
                            "device_info": {
                                "DeviceID": wind_device.DeviceID,
                                "SimpleID": wind_device.SimpleID,
                                "Status": wind_device.Status,
                                "Pout_max": wind_device.Pout_max,
                            },
                            "energy_output": [
                                Energy_optimization_variables_wind_list[index][
                                    ts
                                ].solution_value
                                for ts in range(time_points)
                            ],
                            "regulation_capacity": (
                                [
                                    RegCap_optimization_variables_wind_list[index][
                                        ts
                                    ].solution_value
                                    for ts in range(time_points)
                                ]
                                if participate_regulation
                                else []
                            ),
                            "reserve": (
                                [
                                    Rev_optimization_variables_wind_list[index][
                                        ts
                                    ].solution_value
                                    for ts in range(time_points)
                                ]
                                if participate_reserve
                                else []
                            ),
                        }

                # ESS设备数据
                if Energy_optimization_variables_ess_charge_list and enable_ess:
                    result["device_data"]["ess"] = {}
                    for index, ess_device in enumerate(self.Vpp_temp_devices["ESS"]):
                        device_id = ess_device.SimpleID
                        result["device_data"]["ess"][device_id] = {
                            "device_info": {
                                "DeviceID": ess_device.DeviceID,
                                "SimpleID": ess_device.SimpleID,
                                "Status": ess_device.Status,
                                "Pin_max": ess_device.Pin_max,
                                "Pout_max": ess_device.Pout_max,
                                "Capacity": ess_device.Capacity,
                                "InitialSOC": ess_device.InitialSOC,
                            },
                            "charge_power": [
                                Energy_optimization_variables_ess_charge_list[index][
                                    ts
                                ].solution_value
                                for ts in range(time_points)
                            ],
                            "discharge_power": [
                                Energy_optimization_variables_ess_discharge_list[index][
                                    ts
                                ].solution_value
                                for ts in range(time_points)
                            ],
                            "soc_state": [
                                Energy_optimization_variables_soc_list[index][
                                    ts
                                ].solution_value
                                for ts in range(time_points)
                            ],
                            "charge_binary": [
                                ess_charge_binary_list[index][ts].solution_value
                                for ts in range(time_points)
                            ],
                            "discharge_binary": [
                                ess_discharge_binary_list[index][ts].solution_value
                                for ts in range(time_points)
                            ],
                            "regulation_capacity": (
                                [
                                    RegCap_optimization_variables_ess_list[index][
                                        ts
                                    ].solution_value
                                    for ts in range(time_points)
                                ]
                                if participate_regulation
                                else []
                            ),
                            "reserve": (
                                [
                                    Rev_optimization_variables_ess_list[index][
                                        ts
                                    ].solution_value
                                    for ts in range(time_points)
                                ]
                                if participate_reserve
                                else []
                            ),
                        }

                # DL设备数据
                if Energy_optimization_variables_dl_list and enable_dl:
                    result["device_data"]["dl"] = {}
                    for index, dl_device in enumerate(self.Vpp_temp_devices["DL"]):
                        device_id = dl_device.SimpleID
                        result["device_data"]["dl"][device_id] = {
                            "device_info": {
                                "DeviceID": dl_device.DeviceID,
                                "SimpleID": dl_device.SimpleID,
                                "Status": dl_device.Status,
                                "Pin_max": dl_device.Pin_max,
                                "Pin_min": dl_device.Pin_min,
                                "RequiredDemand": dl_device.RequiredDemand,
                            },
                            "power_input": [
                                Energy_optimization_variables_dl_list[index][
                                    ts
                                ].solution_value
                                for ts in range(time_points)
                            ],
                            "capacity_state": [
                                Capacity_optimization_variables_dl_list[index][
                                    ts
                                ].solution_value
                                for ts in range(time_points)
                            ],
                            "regulation_capacity": (
                                [
                                    RegCap_optimization_variables_dl_list[index][
                                        ts
                                    ].solution_value
                                    for ts in range(time_points)
                                ]
                                if participate_regulation
                                else []
                            ),
                            "reserve": (
                                [
                                    Rev_optimization_variables_dl_list[index][
                                        ts
                                    ].solution_value
                                    for ts in range(time_points)
                                ]
                                if participate_reserve
                                else []
                            ),
                        }

                self.optimization_result = result
                if mode == "intra_day":
                    self._update_states_handler(result, update_point_index=0)
                    print("调用台账更新")
                else:  # 日前No更新
                    # self._update_states_handler(result, update_point_index=0) 
                    print("调用台账更新")
                print("求解完毕")

                return result

            else:
                print("❌ 优化失败: 模型不可行。启动建模诊断分析...")

                # 1. 模型基本结构诊断
                print("\n=== 模型结构诊断 ===")
                print(f"约束数量: {mdl.number_of_constraints}")
                print(f"变量数量: {mdl.number_of_variables}")
                print(f"连续变量数量: {mdl.number_of_continuous_variables}")
                print(f"整数变量数量: {mdl.number_of_integer_variables}")
                print(f"二进制变量数量: {mdl.number_of_binary_variables}")

                # 2. 冲突精炼器深度分析
                try:
                    print("\n=== 冲突精炼器深度分析 ===")
                    cr = ConflictRefiner()
                    conflicts = cr.refine_conflict(mdl)

                    if conflicts:
                        # --- 核心修改在这里 ---
                        # 新版本 docplex 中，conflicts 对象可以直接迭代
                        # 我们将其转换为列表以统一处理
                        try:
                            # 尝试新版本的方式 (直接迭代)
                            conflict_constraints = list(conflicts)
                        except TypeError:
                            # 如果迭代失败，尝试旧版本的方式
                            if hasattr(conflicts, "get_conflict_constraints"):
                                conflict_constraints = (
                                    conflicts.get_conflict_constraints()
                                )
                            else:
                                print(
                                    "警告: 无法从冲突结果中提取约束列表。将直接显示结果。"
                                )
                                conflicts.display()
                                conflict_constraints = []

                        print(f"核心冲突约束: {len(conflict_constraints)}个")
                        conflict_details = []
                        constraint_patterns = {}

                        if not conflict_constraints:
                            # 如果列表为空，也尝试直接打印，因为有些版本可能迭代出空列表但display有内容
                            print("精炼器结果无法转换为列表，尝试直接打印：")
                            conflicts.display()

                        for i, conflict_ct in enumerate(conflict_constraints, 1):
                            # conflict_ct is a _TConflictConstraint(name, element, status)
                            # The element is the actual constraint object from docplex.
                            # Its string representation is the most useful form.
                            display_name = str(conflict_ct.element)
                            conflict_details.append(display_name)
                            print(f"  核心冲突 {i}: {display_name}")

                            # Improved pattern analysis based on the constraint's *own* name if it exists
                            ct_name_attr = getattr(conflict_ct.element, "name", None)
                            if ct_name_attr and "c_" in ct_name_attr:
                                pattern = ct_name_attr.split("_")[1]
                                constraint_patterns[pattern] = (
                                    constraint_patterns.get(pattern, 0) + 1
                                )

                            # 分析冲突类型 (这部分逻辑保持不变)
                            if "energy_balance" in display_name:
                                print("    → 能量平衡约束冲突: 发电与负荷不匹配")
                            elif (
                                "pin_min" in display_name or "pout_min" in display_name
                            ):
                                print("    → 最小功率约束冲突: 设备最小出力限制过严")
                            elif (
                                "pin_max" in display_name or "pout_max" in display_name
                            ):
                                print("    → 最大功率约束冲突: 设备容量限制")
                            elif "work_time" in display_name:
                                print("    → 工作时间约束冲突: 时间窗口设置问题")
                            elif "capacity" in display_name:
                                print("    → 容量约束冲突: 储能或需求响应容量问题")

                        # 3. 约束冲突模式分析 (这部分逻辑保持不变)
                        print(f"\n=== 约束冲突模式分析 ===")
                        if not constraint_patterns:
                            print("未能从冲突约束名称中分析出模式。")
                        else:
                            for pattern, count in constraint_patterns.items():
                                print(f"- {pattern}类约束: {count}个冲突")

                    else:
                        print("冲突精炼器未识别出明确的核心冲突")
                        conflict_details = []
                        constraint_patterns = {}

                except Exception as conflict_error:
                    print(f"冲突精炼器执行失败: {conflict_error}")
                    conflict_details = []
                    constraint_patterns = {}

                print(f"\n=== 建模问题诊断建议 ===")
                suggestions = []

                # 基于冲突约束名称模式提供建议
                if conflict_details:
                    if any("pin_min" in conflict for conflict in conflict_details):
                        suggestions.append(
                            "检查设备Pin_min设置，建议设为0或使用条件约束"
                        )

                    if any(
                        "required_demand" in conflict for conflict in conflict_details
                    ):
                        suggestions.append("降低DL设备RequiredDemand参数(建议500-800)")

                    if any(
                        "initial_capacity" in conflict for conflict in conflict_details
                    ):
                        suggestions.append("调整InitialCapacity为Cap_max的10-30%")

                    if any("work_time" in conflict for conflict in conflict_details):
                        suggestions.append("扩大工作时间窗口(如05:00-23:00)")

                    if any("vpp_output" in conflict for conflict in conflict_details):
                        suggestions.append("检查VPP输出变量下界设置")
                else:
                    # 通用建议
                    suggestions.extend(
                        [
                            "检查设备Pin_min设置，建议设为0或使用条件约束",
                            "检查VPP输出变量下界设置，确保允许负值",
                            "验证设备容量参数设置的合理性",
                            "检查时间窗口约束的设置",
                        ]
                    )

                for i, suggestion in enumerate(suggestions, 1):
                    print(f"{i}. {suggestion}")
                self.optimization_result = None
                # 6. 返回详细诊断结果
                diagnostic_result = {
                    "status": "infeasible",
                    "message": "模型约束冲突，已完成建模诊断",
                    "summary": {
                        "owner_id": owner_id,
                        "time_points": time_points,
                        "mode": mode,
                        "start_ts": start_ts,
                    },
                    "device_selection": {
                        "pv": enable_pv and "PV" in self.Vpp_temp_devices,
                        "wind": enable_wind and "WIND" in self.Vpp_temp_devices,
                        "dg": enable_dg and "DG" in self.Vpp_temp_devices,
                        "ess": enable_ess and "ESS" in self.Vpp_temp_devices,
                        "dl": enable_dl and "DL" in self.Vpp_temp_devices,
                        "tcr": False,
                    },
                    "diagnostics": {
                        "model_structure": {
                            "constraints": mdl.number_of_constraints,
                            "variables": mdl.number_of_variables,
                            "continuous_vars": mdl.number_of_continuous_variables,
                            "integer_vars": mdl.number_of_integer_variables,
                            "binary_vars": mdl.number_of_binary_variables,
                        },
                        "conflict_count": (
                            len(conflict_details) if conflict_details else 0
                        ),
                        "constraint_patterns": constraint_patterns,
                        "core_conflicts": conflict_details,
                        "suggestions": suggestions,
                    },
                    "raw_conflicts": conflict_details,
                }

                return diagnostic_result

        except Exception as e:
            print(f"CPLEX求解器错误详情: {str(e)}")
            mdl.export_as_lp("report_mistake.lp")
            if hasattr(e, "result"):
                print(f"CPLEX求解状态: {e.result.solve_status}")
                print(f"CPLEX详细错误: {e.result.solver_log}")
            print("\n=== 模型结构诊断 ===")
            print(f"约束数量: {mdl.number_of_constraints}")
            print(f"变量数量: {mdl.number_of_variables}")
            print(f"连续变量数量: {mdl.number_of_continuous_variables}")
            print(f"整数变量数量: {mdl.number_of_integer_variables}")
            print(f"二进制变量数量: {mdl.number_of_binary_variables}")
            
            print("❌ 经济调度失败: 模型不可行。启动建模诊断分析...")
            cr = ConflictRefiner()
            conflicts = cr.refine_conflict(mdl)
            if conflicts:
                print("发现以下核心冲突约束:")
                conflicts.display()
                return {"status": "infeasible", "message": "Model is infeasible. Core conflicts identified.",
                        "conflicts": [ str(c.element) for c in conflicts ]}
            else:
                print("冲突精炼器未找到明确的冲突。")
                return {"status": "infeasible",
                        "message": "Model is infeasible, but ConflictRefiner found no specific conflicts."}


    def _update_states_handler(
        self, optimization_result: dict, update_point_index: int = 0
    ):
        """
        状态更新处理器 (Handler)。
        根据优化结果，更新台账(self.vpp_devices)中设备的状态变量。
        这是状态持久化的核心位置。

        参数:
            optimization_result: optimization_for_single_owner 函数返回的完整结果字典。
            update_point_index (int): 使用优化结果中哪个时间点的数据来更新状态,日内15分钟滚动模式下index需要设置为0,因为时间窗口不是一位的.

        返回:
            list: 被更新的设备对象列表
        """
        print("--- [Handler] 开始在内存中更新设备状态 ---")
        updated_devices = []

        if not optimization_result or optimization_result.get("status") != "optimal":
            print("警告：[Handler] 无法更新设备状态，因为没有有效的优化结果。")
            return updated_devices

        owner_id = optimization_result["summary"]["owner_id"]
        device_data = optimization_result.get("device_data", {})
        mode = optimization_result["summary"].get(
            "mode", "day_ahead"
        )  # 获取模式以决定是否打印

        # 更新ESS的current_soc
        if "ess" in device_data:
            for ess_id, ess_results in device_data["ess"].items():
                ess_device_list = [
                    dev
                    for dev in self.vpp_devices.all_devices["ESS"]
                    if dev.SimpleID == ess_id and dev.OwnerID == owner_id
                ]
                if ess_device_list:
                    ess_device = ess_device_list[0]
                    new_soc = ess_results["soc_state"][
                        update_point_index
                    ]  # index should be 0 when utilizing 15min rolling ,do not change
                    ess_device.CurrentSOC = new_soc
                    updated_devices.append(ess_device)
                    print(f"  ESS {ess_id}: 台账中 CurrentSOC 更新为 -> {new_soc:.4f}")

        # 更新DL的cumulative_capacity
        if "dl" in device_data:
            for dl_id, dl_results in device_data["dl"].items():
                dl_device_list = [
                    dev
                    for dev in self.vpp_devices.all_devices["DL"]
                    if dev.SimpleID == dl_id and dev.OwnerID == owner_id
                ]
                if dl_device_list:
                    dl_device = dl_device_list[0]
                    new_capacity = dl_results["capacity_state"][update_point_index]
                    dl_device.cumulative_capacity = new_capacity
                    updated_devices.append(dl_device)
                    print(
                        f"  DL {dl_id}: 台账中 cumulative_capacity 更新为 -> {new_capacity:.4f} kWh"
                    )

        print("--- [Handler] 设备内存状态处理完成 ---\n")
        return updated_devices  # 给外部函数进行文件持久化接口暴露

    def _generate_simple_bidding_curves(
        self,
        optimal_quantities: List[float],
        price_forecasts: List[float],
        min_q_kw: float,
        max_q_kw: float,
        num_segments: int,  # This will be ignored but kept for compatibility
        mode: str,  # This will be ignored
        custom_nonlinear_params: Optional[List[float]] = None,  # Ignored
        price_factor: float = 1.0,  # 新增：价格因子
    ) -> Dict[str, Any]:
        """
        为单个时间段生成简化的、单点的投标曲线。
        价格 = 预测价格 * 价格因子。
        用于替代调频和备用市场的复杂曲线分解，以及单段的电能量市场报价。
        """
        if len(optimal_quantities) != len(price_forecasts):
            raise ValueError("电量列表和价格列表长度必须相同")

        if not optimal_quantities:
            return {"period_results": {}}

        all_period_results = {}
        num_periods = len(optimal_quantities)

        for period_idx in range(num_periods):
            optimal_q = optimal_quantities[period_idx]
            price_forecast = price_forecasts[period_idx]

            # 应用价格因子
            final_price = price_forecast * price_factor

            # 创建一个"退化"的曲线，表示一个点
            # 使用两个非常接近的点来确保绘图函数可以处理
            q1 = max(0.0, optimal_q - 0.001) if "regulation" in mode or "reserve" in mode else optimal_q - 0.001
            q2 = optimal_q
            
            if abs(q2 - q1) < 1e-6:  # 如果 optimal_q 接近于0
                 q1 = optimal_q
                 q2 = optimal_q + 0.001


            quantity_points = np.array([q1, q2])
            price_points = np.array([final_price, final_price])

            curve_points = list(zip(quantity_points, price_points))

            # 创建一个对应的退化段
            segments = [
                {
                    "segment_index": 0,
                    "start_q_kw": quantity_points[0],
                    "end_q_kw": quantity_points[1],
                    "width_kw": quantity_points[1] - quantity_points[0],
                    "price_per_kwh": price_points[1],
                }
            ]

            period_result = {
                "optimal_q_kw": optimal_q,
                "price_forecast_per_kwh": final_price, # 返回应用因子后的价格
                "curve_points": curve_points,
                "segments": segments,
            }
            all_period_results[f"period_{period_idx + 1}"] = period_result

        result = {"period_results": all_period_results}
        return result

    # use for bidding in market
    def bidding_for_market(
        self,
        Bid_segment_number: int = 4,
        mode="uniform",
        custom_nonlinear_params: Optional[List[float]] = None,
    ):
        """
        所生成报量报价段数为输入的self.optimization_result适应段数
        始终使用96点循环价格数据来处理日前和日内模式
        """

        # must be between 1 and 8 unless change the bidding_module.py
        if not (1 <= Bid_segment_number <= 8):
            raise ValueError(f"投标段数必须在1-8之间，当前值: {Bid_segment_number}")
        self.Bid_segment_number = Bid_segment_number

        if (
            self.optimization_result
            and self.optimization_result.get("status") == "optimal"
        ):
            # 1. 从优化结果中提取数据
            vpp_energy_output = self.optimization_result["vpp_data"]["energy_output"]
            vpp_regulation_capacity = self.optimization_result["vpp_data"][
                "regulation_capacity"
            ]
            vpp_reserve = self.optimization_result["vpp_data"]["reserve"]

            summary = self.optimization_result["summary"]
            optimization_mode = summary.get("mode", "day_ahead")
            start_ts = summary.get("start_ts", 0)
            time_points = summary.get("time_points", len(vpp_energy_output))

            # --- 新增：将索引转换为人类可读的时间格式 ---
            # 优化起始时段
            start_time_hours = 0.25 * start_ts
            start_time_minutes = int((start_time_hours % 1) * 60)
            start_time_hours_int = int(start_time_hours) % 24  # 确保小时数在0-23范围内
            start_period_str = f"{start_time_hours_int:02d}:{start_time_minutes:02d}"

            # 优化结束时段 (最后一个时段的结束时间)
            end_ts_index = start_ts + time_points  # 修正：不减1，因为这是结束时间
            end_time_hours = 0.25 * end_ts_index
            end_time_minutes = int((end_time_hours % 1) * 60)
            end_time_hours_int = int(end_time_hours) % 24  # 确保小时数在0-23范围内
            end_period_str = f"{end_time_hours_int:02d}:{end_time_minutes:02d}"

            print(f"\n--- 投标时段分析 ---")
            print(f"模式: '{optimization_mode}'")
            print(f"真实优化起始时段: {start_period_str} (索引: {start_ts})")
            print(f"时段覆盖: {start_period_str} ~ {end_period_str}")
            print(f"总计: {time_points}个15分钟时段")
            print("-" * 22)
            self.info_for_ploting = f"模式: '{optimization_mode}' 真实优化时段覆盖: {start_period_str} ~ {end_period_str} 总计: {time_points}个15分钟时段"

            # 2. 核心逻辑: 始终使用循环索引(%)获取价格数据窗口
            total_price_points = len(self.EnergyMarketPrice)  # 应为96

            # 检查价格数据是否存在，如果不存在或为空则设为空列表
            energy_price_window = (
                [
                    self.EnergyMarketPrice[(start_ts + i) % total_price_points]
                    for i in range(time_points)
                ]
                if hasattr(self, "EnergyMarketPrice") and self.EnergyMarketPrice
                else []
            )
            regcap_price_window = (
                [
                    self.RegCapMarketPrice[(start_ts + i) % total_price_points]
                    for i in range(time_points)
                ]
                if hasattr(self, "RegCapMarketPrice") and self.RegCapMarketPrice
                else []
            )
            reserve_price_window = (
                [
                    self.ReserveMarketPrice[(start_ts + i) % total_price_points]
                    for i in range(time_points)
                ]
                if hasattr(self, "ReserveMarketPrice") and self.ReserveMarketPrice
                else []
            )

            # print(f"模式: {optimization_mode}, 起始点: {start_ts}. 已生成长度为 {len(energy_price_window)} 的循环价格窗口。")

            # 3. 为每个市场生成投标曲线
            print("\n--- 开始为各市场生成投标曲线 ---")

            # 电能量市场
            if self.Bid_segment_number == 1:
                print("✓ 生成电能量市场单点投标 (分段数=1)")
                vpp_energy_bidding_result = self._generate_simple_bidding_curves(
                    optimal_quantities=vpp_energy_output,
                    price_forecasts=energy_price_window,
                    min_q_kw=self.vpp_available_Quantity_min,
                    max_q_kw=self.vpp_available_Quantity_max,
                    num_segments=1,
                    mode="energy",
                    price_factor=1.0,  # 电能量市场单点报价，因子为1
                )
            else:
                print(f"✓ 生成电能量市场多段投标曲线 (分段数={self.Bid_segment_number})")
                vpp_energy_bidding_result = generate_multi_period_bidding_curves(
                    optimal_quantities=vpp_energy_output,
                    price_forecasts=energy_price_window,
                    min_q_kw=self.vpp_available_Quantity_min,
                    max_q_kw=self.vpp_available_Quantity_max,
                    num_segments=self.Bid_segment_number,
                    mode=mode,
                    custom_nonlinear_params=custom_nonlinear_params,
                )

            # 调频容量市场
            vpp_regulation_bidding_result = None
            if vpp_regulation_capacity and any(
                cap > 1e-6 for cap in vpp_regulation_capacity
            ):
                print("✓ 生成调频容量市场单点投标")
                vpp_regulation_bidding_result = self._generate_simple_bidding_curves(
                    optimal_quantities=vpp_regulation_capacity,
                    price_forecasts=regcap_price_window,
                    min_q_kw=0,  # 调频容量最小为0
                    max_q_kw=self.vpp_available_RegCap_max,
                    num_segments=1,
                    mode="regulation",
                    price_factor=0.8,  # 按要求，辅助服务使用0.8的因子
                )
            else:
                print("✗ 跳过调频容量市场投标曲线生成 (数据全为0或无数据)")

            # 备用市场
            vpp_reserve_bidding_result = None
            if vpp_reserve and any(res > 1e-6 for res in vpp_reserve):
                print("✓ 生成备用市场单点投标")
                vpp_reserve_bidding_result = self._generate_simple_bidding_curves(
                    optimal_quantities=vpp_reserve,
                    price_forecasts=reserve_price_window,
                    min_q_kw=0,
                    max_q_kw=self.vpp_available_Rev_max,
                    num_segments=1,
                    mode="reserve",
                    price_factor=0.8,  # 按要求，辅助服务使用0.8的因子
                )
            else:
                print("✗ 跳过备用市场投标曲线生成 (无可用备用容量)")

            # 4. 汇总所有市场的投标结果
            self.bidding_segment_useful_curves = {
                "energy": vpp_energy_bidding_result,
                "regulation_capacity": vpp_regulation_bidding_result,
                "reserve": vpp_reserve_bidding_result,
            }

            # 5. 构造并返回结果字典
            bidding_result = {
                "status": "success",
                "mode": mode,
                "optimization_mode": optimization_mode,
                "start_ts": start_ts,
                "owner_id": self.optimization_result["summary"]["owner_id"],
                "time_points": self.optimization_result["summary"]["time_points"],
                "bid_segment_number": self.Bid_segment_number,
                "bidding_segment_useful_curves": self.bidding_segment_useful_curves,
            }
            self.bidding_result = bidding_result
            return bidding_result
            
        else:
            print("警告：没有可用的优化结果，无法生成投标曲线")
            bidding_result = {
                "status": "failed",
                "message": "没有可用的优化结果",
                "bidding_segment_useful_curves": {},
            }
            self.bidding_result = bidding_result
            return bidding_result

    def bidding_result_plot(
        self,
        output_path: str,
        market_key: str,
        points_to_plot=None,
        focus_on_generation=False,
        focus_on_optimal=False,
        focus_width_mw=2.0,
    ):
        """
        绘制每个时段的投标曲线，现在将 focus_on_optimal 作为主要功能。
        它会绘制总的供给曲线和需求曲线，并在每个时间点上标记出最优运行点。

        :param output_path: (str) 保存图像的文件路径。
        :param market_key: (str) 需要绘制的市场的键名 (e.g., 'energy', 'regulation_capacity').
        :param points_to_plot: (list, optional) 指定要绘制的时间点索引列表。如果为 None，则绘制所有时间点。
        :param focus_on_generation: (bool) 如果为True，则Y轴将关注发电（正值）。
        :param focus_on_optimal: (bool) 如果为True，则图表将放大到最优出清点附近。
        :param focus_width_mw: (float) 当 focus_on_optimal 为True时，围绕最优点的放大宽度（兆瓦）。
        """
        print(
            f"🔍 DEBUG: 开始为市场 '{market_key}' 绘制投标曲线图，输出路径: {output_path}"
        )

        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime, timedelta

        # 设置字体和样式 - 使用英文避免字体问题
        plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # 尝试使用可用的样式，如果不可用则使用默认样式
        try:
            # 首先尝试新版本的seaborn样式
            if "seaborn-v0_8-darkgrid" in plt.style.available:
                plt.style.use("seaborn-v0_8-darkgrid")
            elif "seaborn-darkgrid" in plt.style.available:
                plt.style.use("seaborn-darkgrid")
            elif "seaborn" in plt.style.available:
                plt.style.use("seaborn")
            else:
                # 使用默认样式并手动设置网格
                plt.style.use("default")
                plt.rcParams["axes.grid"] = True
                plt.rcParams["grid.alpha"] = 0.3
        except:
            # 如果样式设置失败，使用默认样式
            plt.style.use("default")
            plt.rcParams["axes.grid"] = True
            plt.rcParams["grid.alpha"] = 0.3

        # 设置更美观的默认参数
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "#FAFAFA"
        plt.rcParams["axes.edgecolor"] = "#CCCCCC"
        plt.rcParams["axes.linewidth"] = 1.2
        plt.rcParams["grid.linestyle"] = "--"
        plt.rcParams["grid.linewidth"] = 0.8

        print(f"🔍 DEBUG: 检查投标结果数据...")
        if not self.bidding_result or self.bidding_result.get("status") != "success":
            print("⚠️  Warning: No available bidding result data, cannot plot charts")
            return None

        bidding_curves = self.bidding_result.get("bidding_segment_useful_curves")
        if not bidding_curves or market_key not in bidding_curves:
            print(f"⚠️ Warning: 在投标结果中未找到市场 '{market_key}' 的数据。")
            return None

        # 如果没有指定绘制点数，使用全天时间点
        if points_to_plot is None:
            points_to_plot = VirtualPowerPlantHost.time_dayahead_points
        plot_n = min(points_to_plot, len(bidding_curves[market_key]["period_results"]))
        print(f"🔍 DEBUG: 将绘制 {plot_n} 个时间点")

        # 市场名称映射和颜色配置 - 全英文
        markets = {
            "energy": {"name": "Energy Market", "icon": "⚡", "color": "#2E8B57"},
            "regulation_capacity": {
                "name": "Regulation Capacity Market",
                "icon": "⚙️",
                "color": "#4169E1",
            },
            "reserve": {"name": "Reserve Market", "icon": "🔒", "color": "#FF6347"},
        }

        if market_key not in markets:
            print(f"⚠️ Warning: 未知的市场键名 '{market_key}'。")
            return None

        market_config = markets[market_key]
        market_data = bidding_curves[market_key]

        print(
            f"📊 Plotting {market_config['icon']} {market_config['name']} bidding curves..."
        )

        period_results = market_data["period_results"]

        Price_curve = np.full([self.Bid_segment_number, plot_n], np.nan)
        Quantity_curve_kw = np.full([self.Bid_segment_number + 1, plot_n], np.nan)

        for t in range(plot_n):
            period_key = f"period_{t + 1}"
            if period_key in period_results:
                period_data = period_results[period_key]
                curve_points = period_data["curve_points"]
                segments = period_data["segments"]

                if curve_points:
                    # 提取量价数据
                    q_kw, p = zip(*curve_points)
                    Quantity_curve_kw[: len(q_kw), t] = q_kw
                    Price_curve[: len(p) - 1, t] = p[:-1]

        # --- 单位转换: kW -> MW ---
        Quantity_curve_mw = Quantity_curve_kw / 1000.0
        optimal_quantities_mw = self.bidding_result.get("optimal_quantities", {})
        optimal_quantities_mw = {
            k: np.array(v) / 1000.0 for k, v in optimal_quantities_mw.items()
        }

        # 
        print(f" DEBUG: 创建图形对象...")
        fig = plt.figure(figsize=(18, 14), facecolor="white")
        fig.patch.set_facecolor("white")
        print(f" DEBUG: 图形对象创建成功")

        # 构建简洁清晰的标题
        mode_text = (
            "Day-ahead"
            if self.optimization_result["summary"]["mode"] == "day_ahead"
            else "Intraday"
        )
        start_ts = self.optimization_result["summary"]["start_ts"]
        end_ts = start_ts + plot_n
        start_hour = (start_ts // 4) % 24
        end_hour = (end_ts // 4) % 24
        start_minute = (start_ts % 4) * 15
        end_minute = (end_ts % 4) * 15

        # 主标题 - 简洁版
        main_title = f"VPP {market_config['name']} Bidding Curves"

        # 副标题 - 重新组织，更清晰
        subtitle = f"User: {self.optimization_result['summary']['owner_id']} | {mode_text} Mode | {start_hour:02d}:{start_minute:02d}-{end_hour:02d}:{end_minute:02d} | {plot_n} Periods"

        # 设置主标题样式
        fig.suptitle(
            main_title,
            fontsize=18,
            fontweight="bold",
            color=market_config["color"],
            y=0.95,
        )

        # 设置副标题
        fig.text(
            0.5,
            0.91,
            subtitle,
            ha="center",
            va="top",
            fontsize=11,
            color="#2F4F4F",
            style="normal",
        )

        # 动态计算子图布局
        if plot_n <= 4:
            rows, cols = 2, 2
        elif plot_n <= 6:
            rows, cols = 2, 3
        elif plot_n <= 8:
            rows, cols = 2, 4
        elif plot_n <= 12:
            rows, cols = 3, 4
        elif plot_n <= 16:
            rows, cols = 4, 4
        elif plot_n <= 24:
            rows, cols = 4, 6
        elif plot_n <= 48:
            rows, cols = 6, 8
        else:
            rows, cols = 8, 12

        # 计算所有时段的全局最大最小值，用于统一坐标轴范围
        global_q_min, global_q_max = float('inf'), float('-inf')
        global_p_min, global_p_max = float('inf'), float('-inf')
        
        for t in range(plot_n):
            q_mw_all = Quantity_curve_mw[:, t]
            p_all = Price_curve[:, t]
            valid_q = q_mw_all[~np.isnan(q_mw_all)]
            valid_p = p_all[~np.isnan(p_all)]
            if len(valid_q) > 0:
                global_q_min = min(global_q_min, np.min(valid_q))
                global_q_max = max(global_q_max, np.max(valid_q))
            if len(valid_p) > 0:
                global_p_min = min(global_p_min, np.min(valid_p))
                global_p_max = max(global_p_max, np.max(valid_p))
        
        # 计算全局边距
        global_q_margin = (global_q_max - global_q_min) * 0.05 if (global_q_max - global_q_min) > 1e-6 else 0.1
        global_p_margin = (global_p_max - global_p_min) * 0.05 if (global_p_max - global_p_min) > 1e-6 else 1

        for t in range(plot_n):
            ax = fig.add_subplot(rows, cols, t + 1)

            q_mw_all = Quantity_curve_mw[:, t]
            p_all = Price_curve[:, t]

            valid_p_mask = ~np.isnan(p_all)
            if not np.any(valid_p_mask):
                continue

            q_mw_valid = Quantity_curve_mw[~np.isnan(Quantity_curve_mw[:, t]), t]
            p_valid = p_all[valid_p_mask]

            # 所有子图使用相同的坐标轴范围
            ax.set_xlim(global_q_min - global_q_margin, global_q_max + global_q_margin)
            ax.set_ylim(global_p_min - global_p_margin, global_p_max + global_p_margin)

            # 绘制美化的阶梯曲线
            ax.step(
                q_mw_valid,
                np.append(p_valid, p_valid[-1]),
                where="post",
                linewidth=2.5,
                color=market_config["color"],
                alpha=0.8,
                label="Bidding Curve",
            )

            # 绘制投标点
            ax.scatter(
                q_mw_valid,
                np.append(p_valid, p_valid[-1]),
                color="#DC143C",
                s=25,
                alpha=0.8,
                zorder=5,
                edgecolor="white",
                linewidth=1,
            )

            # 美化子图标题 - 显示实际计算时间
            global_ts = start_ts + t  # 计算全局时间索引
            cyclic_ts = (
                global_ts % VirtualPowerPlantHost.time_dayahead_points
            )  # 循环到96点内
            actual_hour = (cyclic_ts // 4) % 24
            actual_minute = (cyclic_ts % 4) * 15
            period_title = f"T{t + 1}\n{actual_hour:02d}:{actual_minute:02d}"
            ax.set_title(
                period_title, fontsize=10, fontweight="bold", color="#2F4F4F", pad=10
            )

            # 设置坐标轴标签
            if t % cols == 0:
                ax.set_ylabel(
                    "Price (CNY/MWh)", fontsize=9, fontweight="bold", color="#2F4F4F"
                )
            if t >= plot_n - cols:
                ax.set_xlabel(
                    "Power (MW)", fontsize=9, fontweight="bold", color="#2F4F4F"
                )

            # 美化坐标轴
            ax.tick_params(labelsize=8, colors="#2F4F4F")

        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(
            top=0.86, bottom=0.08, left=0.06, right=0.96, hspace=0.35, wspace=0.25
        )

        # 添加图例（仅在第一个子图中显示）
        if plot_n > 0:
            axes = fig.get_axes()
            if axes:
                axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)

        # 最终保存
        try:
            plt.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
            print(f"✅ 图像已成功保存到: {output_path}")
        except Exception as e:
            print(f"❌ 保存图像失败: {e}")
        finally:
            plt.close(fig)

        return {"status": "success", "path": output_path}

    def economic_dispatch(
            self,
            owner_id: str,
            cleared_contracts: Dict[ str, List[ float ] ],
            market_and_load_data: Dict[ str, List[ float ] ],
            time_points: int,
            start_ts: int = 0,
            enable_pv=True,
            enable_wind=True,
            enable_dg=True,
            enable_ess=True,
            enable_dl=True,
            penalty_factor_energy=1000,
            penalty_factor_ancillary=800,
    ):
        """
        【最终完整版】经济调度核心函数，根据已中标的合同曲线，对VPP内各类设备进行功率分解。
        此版本包含了与优化函数同等级别的、完整且正确的设备物理建模。
        """
        # =================================================================
        # PART 1: PREPARE DATA and PARAMETERS
        # =================================================================
        print(f"INFO: economic_dispatch received cleared_contracts for markets: {list(cleared_contracts.keys())}")
        print(f"INFO:cleared_contracts: {cleared_contracts.values()}")
        print(f"INFO:market_and_load_data: {market_and_load_data}")
        
        self.BusLoad = market_and_load_data.get("busload", [ 0 ] * time_points)
        self.Vpp_temp_devices = self.vpp_devices._get_devices_by_owner(owner_id)
        mdl = Model(name=f"VPP_Economic_Dispatch_Final_{owner_id}")

        # =================================================================
        # PART 2: CREATE FULL DECISION VARIABLES
        # =================================================================
        # --- PV ---
        Energy_optimization_variables_pv_list, RegCap_optimization_variables_pv_list, Rev_optimization_variables_pv_list = (
            [ ], [ ], [ ])
        if enable_pv and self.Vpp_temp_devices.get("PV"):
            for pv_unit in self.Vpp_temp_devices[ "PV" ]:
                Energy_optimization_variables_pv_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"Energy_PV_{pv_unit.SimpleID}_Pout", lb=0))
                RegCap_optimization_variables_pv_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"RegCap_PV_{pv_unit.SimpleID}_Pout", lb=0))
                Rev_optimization_variables_pv_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"Rev_PV_{pv_unit.SimpleID}_Pout", lb=0))
        # --- WIND ---
        Energy_optimization_variables_wind_list, RegCap_optimization_variables_wind_list, Rev_optimization_variables_wind_list = (
            [ ], [ ], [ ])
        if enable_wind and self.Vpp_temp_devices.get("WIND"):
            for wind_unit in self.Vpp_temp_devices[ "WIND" ]:
                Energy_optimization_variables_wind_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"Energy_WIND_{wind_unit.SimpleID}_Pout", lb=0))
                RegCap_optimization_variables_wind_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"RegCap_WIND_{wind_unit.SimpleID}_Pout", lb=0))
                Rev_optimization_variables_wind_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"Rev_WIND_{wind_unit.SimpleID}_Pout", lb=0))
        # --- DG ---
        Energy_optimization_variables_dg_list, RegCap_optimization_variables_dg_list, Rev_optimization_variables_dg_list = (
            [ ], [ ], [ ])
        if enable_dg and self.Vpp_temp_devices.get("DG"):
            for dg_unit in self.Vpp_temp_devices[ "DG" ]:
                Energy_optimization_variables_dg_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"Energy_DG_{dg_unit.SimpleID}_Pout", lb=0))
                RegCap_optimization_variables_dg_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"RegCap_DG_{dg_unit.SimpleID}_Pout", lb=0))
                Rev_optimization_variables_dg_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"Rev_DG_{dg_unit.SimpleID}_Pout", lb=0))
        # --- ESS ---
        Energy_optimization_variables_ess_charge_list, Energy_optimization_variables_ess_discharge_list, RegCap_optimization_variables_ess_list, Rev_optimization_variables_ess_list, Energy_optimization_variables_soc_list, ess_charge_binary_list, ess_discharge_binary_list = (
            [ ], [ ], [ ], [ ], [ ], [ ], [ ])
        if enable_ess and self.Vpp_temp_devices.get("ESS"):
            for ess_unit in self.Vpp_temp_devices[ "ESS" ]:
                Energy_optimization_variables_ess_charge_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"Energy_ESS_{ess_unit.SimpleID}_Pin", lb=0))
                Energy_optimization_variables_ess_discharge_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"Energy_ESS_{ess_unit.SimpleID}_Pout", lb=0))
                RegCap_optimization_variables_ess_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"RegCap_ESS_{ess_unit.SimpleID}", lb=0))
                Rev_optimization_variables_ess_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"Rev_ESS_{ess_unit.SimpleID}", lb=0))
                Energy_optimization_variables_soc_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"SOC_ESS_{ess_unit.SimpleID}", lb=0, ub=1))
                ess_charge_binary_list.append(
                    mdl.binary_var_list(keys=time_points, name=f"ESS_{ess_unit.SimpleID}_charge_binary"))
                ess_discharge_binary_list.append(
                    mdl.binary_var_list(keys=time_points, name=f"ESS_{ess_unit.SimpleID}_discharge_binary"))
        # --- DL ---
        Capacity_optimization_variables_dl_list, Energy_optimization_variables_dl_list, RegCap_optimization_variables_dl_list, Rev_optimization_variables_dl_list = (
            [ ], [ ], [ ], [ ])
        if enable_dl and self.Vpp_temp_devices.get("DL"):
            
            
            
            
            
               # debug
            print("--- 检查DL设备关键参数 ---")
            for dl_device in self.Vpp_temp_devices["DL"]:
                print(f"设备ID: {dl_device.SimpleID}")
                print(f"  - RequiredDemand: {dl_device.RequiredDemand}")
                print(f"  - InitialCapacity: {dl_device.InitialCapacity}")
                print(f"  - Cap_max: {dl_device.Cap_max}")
                print(f"  - Pin_max: {dl_device.Pin_max}")
                print(f"  - StartTime: {dl_device.StartTime}")
                print(f"  - EndTime: {dl_device.EndTime}")
            print("--------------------------")
            # end
    
            
            
            
            
            
            
            
            for dl_unit in self.Vpp_temp_devices[ "DL" ]:
                Capacity_optimization_variables_dl_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"Capacity_DL_{dl_unit.SimpleID}",
                                            lb=dl_unit.Cap_min, ub=dl_unit.Cap_max))
                Energy_optimization_variables_dl_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"Energy_DL_{dl_unit.SimpleID}_Pin", lb=0,
                                            ub=dl_unit.Pin_max))
                RegCap_optimization_variables_dl_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"RegCap_DL_{dl_unit.SimpleID}_Pin", lb=0))
                Rev_optimization_variables_dl_list.append(
                    mdl.continuous_var_list(keys=time_points, name=f"Rev_DL_{dl_unit.SimpleID}_Pin", lb=0))

        # =================================================================
        # PART 3: ADD FULL & CORRECT DEVICE-LEVEL CONSTRAINTS
        # =================================================================
        for ts in range(time_points):
            time_interval = 0.25

            # --- DG 设备约束 ---
            if enable_dg and self.Vpp_temp_devices.get("DG"):
                for index, dg_device in enumerate(self.Vpp_temp_devices[ "DG" ]):
                    energy_limit = dg_device.get_bid_limit_energy_max()
                    reg_cap_limit = dg_device.get_bid_limit_reg_cap_max()
                    rev_limit = dg_device.get_bid_limit_rev_max()

                    mdl.add_constraint(Energy_optimization_variables_dg_list[ index ][ ts ] <= energy_limit)
                    mdl.add_constraint(Energy_optimization_variables_dg_list[ index ][ ts ] >= dg_device.Pout_min)
                    mdl.add_constraint(RegCap_optimization_variables_dg_list[ index ][ ts ] <= reg_cap_limit)
                    mdl.add_constraint(RegCap_optimization_variables_dg_list[ index ][ ts ] >= 0)
                    mdl.add_constraint(Rev_optimization_variables_dg_list[ index ][ ts ] <= rev_limit)
                    mdl.add_constraint(Rev_optimization_variables_dg_list[ index ][ ts ] >= 0)

                    mdl.add_constraint(Energy_optimization_variables_dg_list[ index ][ ts ] +
                                       RegCap_optimization_variables_dg_list[ index ][ ts ] +
                                       Rev_optimization_variables_dg_list[ index ][ ts ] <= dg_device.Pout_max)
                    mdl.add_constraint(Energy_optimization_variables_dg_list[ index ][ ts ] -
                                       RegCap_optimization_variables_dg_list[ index ][ ts ] >= dg_device.Pout_min)

            # --- PV 设备约束 ---
            if enable_pv and self.Vpp_temp_devices.get("PV"):
                for index, pv_device in enumerate(self.Vpp_temp_devices[ "PV" ]):
                    forecast = pv_device.get_forecast_power()
                    physical_limit = forecast[ ts ] if forecast and ts < len(forecast) else pv_device.Pout_max
                    reg_cap_limit = pv_device.get_bid_limit_reg_cap_max()
                    rev_limit = pv_device.get_bid_limit_rev_max()

                    mdl.add_constraint(RegCap_optimization_variables_pv_list[ index ][ ts ] <= reg_cap_limit)
                    mdl.add_constraint(Rev_optimization_variables_pv_list[ index ][ ts ] <= rev_limit)

                    mdl.add_constraint(Energy_optimization_variables_pv_list[ index ][ ts ] +
                                       RegCap_optimization_variables_pv_list[ index ][ ts ] +
                                       Rev_optimization_variables_pv_list[ index ][ ts ] <= physical_limit)
                    mdl.add_constraint(Energy_optimization_variables_pv_list[ index ][ ts ] -
                                       RegCap_optimization_variables_pv_list[ index ][ ts ] >= 0)

                    mdl.add_constraint(Energy_optimization_variables_pv_list[ index ][ ts ] >= 0)
                    mdl.add_constraint(RegCap_optimization_variables_pv_list[ index ][ ts ] >= 0)
                    mdl.add_constraint(Rev_optimization_variables_pv_list[ index ][ ts ] >= 0)

            # --- WIND 设备约束 ---
            if enable_wind and self.Vpp_temp_devices.get("WIND"):
                for index, wind_device in enumerate(self.Vpp_temp_devices[ "WIND" ]):
                    forecast = wind_device.get_forecast_power()
                    physical_limit = forecast[ ts ] if forecast and ts < len(forecast) else wind_device.Pout_max
                    reg_cap_limit = wind_device.get_bid_limit_reg_cap_max()
                    rev_limit = wind_device.get_bid_limit_rev_max()

                    mdl.add_constraint(RegCap_optimization_variables_wind_list[ index ][ ts ] <= reg_cap_limit)
                    mdl.add_constraint(Rev_optimization_variables_wind_list[ index ][ ts ] <= rev_limit)

                    mdl.add_constraint(Energy_optimization_variables_wind_list[ index ][ ts ] +
                                       RegCap_optimization_variables_wind_list[ index ][ ts ] +
                                       Rev_optimization_variables_wind_list[ index ][ ts ] <= physical_limit)
                    mdl.add_constraint(Energy_optimization_variables_wind_list[ index ][ ts ] -
                                       RegCap_optimization_variables_wind_list[ index ][ ts ] >= 0)

                    mdl.add_constraint(Energy_optimization_variables_wind_list[ index ][ ts ] >= 0)
                    mdl.add_constraint(RegCap_optimization_variables_wind_list[ index ][ ts ] >= 0)
                    mdl.add_constraint(Rev_optimization_variables_wind_list[ index ][ ts ] >= 0)

            # --- ESS 设备约束 ---
            if enable_ess and self.Vpp_temp_devices.get("ESS"):
                for index, ess_device in enumerate(self.Vpp_temp_devices[ "ESS" ]):
                    mdl.add_constraint(
                        Energy_optimization_variables_ess_charge_list[ index ][ ts ] <= ess_device.Pin_max)
                    mdl.add_constraint(
                        Energy_optimization_variables_ess_discharge_list[ index ][ ts ] <= ess_device.Pout_max)
                    mdl.add_constraint(
                        RegCap_optimization_variables_ess_list[ index ][ ts ] <= ess_device.get_bid_limit_reg_cap_max())
                    mdl.add_constraint(
                        Rev_optimization_variables_ess_list[ index ][ ts ] <= ess_device.get_bid_limit_rev_max())

                    mdl.add_constraint(Energy_optimization_variables_ess_discharge_list[ index ][ ts ] +
                                       RegCap_optimization_variables_ess_list[ index ][ ts ] +
                                       Rev_optimization_variables_ess_list[ index ][ ts ] <= ess_device.Pout_max)
                    mdl.add_constraint(Energy_optimization_variables_ess_charge_list[ index ][ ts ] +
                                       RegCap_optimization_variables_ess_list[ index ][ ts ] <= ess_device.Pin_max)

                    mdl.add_constraint(
                        ess_charge_binary_list[ index ][ ts ] + ess_discharge_binary_list[ index ][ ts ] <= 1)
                    mdl.add_constraint(
                        Energy_optimization_variables_ess_charge_list[ index ][ ts ] <= ess_charge_binary_list[ index ][
                            ts ] * ess_device.Pin_max)
                    mdl.add_constraint(Energy_optimization_variables_ess_discharge_list[ index ][ ts ] <=
                                       ess_discharge_binary_list[ index ][ ts ] * ess_device.Pout_max)

                    mdl.add_constraint(Energy_optimization_variables_soc_list[ index ][ ts ] <= 0.98)
                    mdl.add_constraint(Energy_optimization_variables_soc_list[ index ][ ts ] >= 0.05)
                    eff, eff_inv = ess_device.TransferEfficiency, 1.0 / ess_device.TransferEfficiency if abs(
                        ess_device.TransferEfficiency) > 1e-6 else 1e6
                    if ts == 0:
                        start_soc = getattr(ess_device, "CurrentSOC", ess_device.InitialSOC)
                        mdl.add_constraint(Energy_optimization_variables_soc_list[ index ][
                                               ts ] * ess_device.Capacity == start_soc * ess_device.Capacity * ess_device.SelfDischargeRate + (
                                                       Energy_optimization_variables_ess_charge_list[ index ][
                                                           ts ] * eff -
                                                       Energy_optimization_variables_ess_discharge_list[ index ][
                                                           ts ] * eff_inv) * time_interval)
                    else:
                        mdl.add_constraint(
                            Energy_optimization_variables_soc_list[ index ][ ts ] * ess_device.Capacity ==
                            Energy_optimization_variables_soc_list[ index ][
                                ts - 1 ] * ess_device.Capacity * ess_device.SelfDischargeRate + (
                                        Energy_optimization_variables_ess_charge_list[ index ][ ts ] * eff -
                                        Energy_optimization_variables_ess_discharge_list[ index ][
                                            ts ] * eff_inv) * time_interval)

            # --- DL 设备约束 ---
            if enable_dl and self.Vpp_temp_devices.get("DL"):
                for index, dl_device in enumerate(self.Vpp_temp_devices[ "DL" ]):
                    mdl.add_constraint(
                        RegCap_optimization_variables_dl_list[ index ][ ts ] <= dl_device.get_bid_limit_reg_cap_max())
                    mdl.add_constraint(
                        Rev_optimization_variables_dl_list[ index ][ ts ] <= dl_device.get_bid_limit_rev_max())

                    mdl.add_constraint(Energy_optimization_variables_dl_list[ index ][ ts ] +
                                       RegCap_optimization_variables_dl_list[ index ][ ts ] +
                                       Rev_optimization_variables_dl_list[ index ][ ts ] <= dl_device.Pin_max)
                    mdl.add_constraint(Energy_optimization_variables_dl_list[ index ][ ts ] -
                                       RegCap_optimization_variables_dl_list[ index ][ ts ] >= dl_device.Pin_min)

                    global_ts = start_ts + ts
                    current_hour = (global_ts // 4) % 24
                    current_minute = (global_ts % 4) * 15
                    current_time_minutes = current_hour * 60 + current_minute
                    work_start = dl_device.StartTime;
                    work_end = dl_device.EndTime
                    work_start_minutes = work_start.hour * 60 + work_start.minute
                    work_end_minutes = work_end.hour * 60 + work_end.minute
                    is_work_time = (
                                work_start_minutes <= current_time_minutes < work_end_minutes) if work_end_minutes > work_start_minutes else (
                                (current_time_minutes >= work_start_minutes) or (
                                    current_time_minutes < work_end_minutes))

                    if is_work_time:
                        mdl.add_constraint(Energy_optimization_variables_dl_list[ index ][ ts ] >= dl_device.Pin_min)
                    else:
                        mdl.add_constraint(Energy_optimization_variables_dl_list[ index ][ ts ] == 0)

                    if ts == 0:
                        start_capacity = getattr(dl_device, "cumulative_capacity", dl_device.InitialCapacity*dl_device.Cap_max)
                        mdl.add_constraint(Capacity_optimization_variables_dl_list[ index ][ ts ] == start_capacity +
                                           Energy_optimization_variables_dl_list[ index ][ ts ] * time_interval)
                    else:
                        mdl.add_constraint(Capacity_optimization_variables_dl_list[ index ][ ts ] ==
                                           Capacity_optimization_variables_dl_list[ index ][ ts - 1 ] +
                                           Energy_optimization_variables_dl_list[ index ][ ts ] * time_interval)
                    mdl.add_constraint(Capacity_optimization_variables_dl_list[ index ][ ts ] <= dl_device.Cap_max)
                    mdl.add_constraint(Capacity_optimization_variables_dl_list[ index ][ ts ] >= dl_device.Cap_min)

        # # --- DL 任务完成约束 ---
        # if enable_dl and self.Vpp_temp_devices.get("DL"):
        #     for index, dl_device in enumerate(self.Vpp_temp_devices[ "DL" ]):
        #         if dl_device.RequiredDemand > 0:
        #             pass
        #             mdl.add_constraint(Capacity_optimization_variables_dl_list[ index ][
        #                                    time_points - 1 ] >= dl_device.RequiredDemand * 0.99)

        # --- DL 任务完成软约束 ---
        dl_shortfall_vars = [ ]  # 用于在目标函数中收集所有缺口变量
        if enable_dl and self.Vpp_temp_devices.get("DL"):
            for index, dl_device in enumerate(self.Vpp_temp_devices[ "DL" ]):
                if dl_device.RequiredDemand > 0:
                    # 1. 定义一个代表"任务缺口"的连续变量，单位 kWh，下限为0
                    shortfall_var = mdl.continuous_var(name=f"dl_shortfall_{dl_device.SimpleID}", lb=0)
                    dl_shortfall_vars.append(shortfall_var)

                    # 2. 添加新的软约束：
                    #    期末累计容量 + 任务缺口 >= 总需求
                    #    这个约束总是可行的，因为 shortfall_var 可以取任意大的正值
                    mdl.add_constraint(
                        Capacity_optimization_variables_dl_list[ index ][ time_points - 1 ] + shortfall_var
                        >= dl_device.RequiredDemand * 0.99,
                        ctname=f"c_dl_soft_demand_meet_{dl_device.SimpleID}"
                    )

        # =================================================================
        # PART 4: VPP AGGREGATION & DEVIATION VARIABLES (TypeError FIX)
        # =================================================================
        energy_pos_dev = mdl.continuous_var_list(keys=time_points, name="pdev_energy", lb=0)
        energy_neg_dev = mdl.continuous_var_list(keys=time_points, name="ndev_energy", lb=0)
        reg_cap_pos_dev = mdl.continuous_var_list(keys=time_points, name="pdev_reg_cap", lb=0)
        reg_cap_neg_dev = mdl.continuous_var_list(keys=time_points, name="ndev_reg_cap", lb=0)
        rev_pos_dev = mdl.continuous_var_list(keys=time_points, name="pdev_reserve", lb=0)
        rev_neg_dev = mdl.continuous_var_list(keys=time_points, name="ndev_reserve", lb=0)

        for ts in range(time_points):
            # --- 建立当前时间步(ts)的所有项的列表 ---
            energy_terms = [ ]
            reg_cap_terms = [ ]
            reserve_terms = [ ]

            if enable_pv and self.Vpp_temp_devices.get("PV"):
                for i in range(len(self.Vpp_temp_devices[ "PV" ])):
                    energy_terms.append(Energy_optimization_variables_pv_list[ i ][ ts ])
                    reg_cap_terms.append(RegCap_optimization_variables_pv_list[ i ][ ts ])
                    reserve_terms.append(Rev_optimization_variables_pv_list[ i ][ ts ])

            if enable_wind and self.Vpp_temp_devices.get("WIND"):
                for i in range(len(self.Vpp_temp_devices[ "WIND" ])):
                    energy_terms.append(Energy_optimization_variables_wind_list[ i ][ ts ])
                    reg_cap_terms.append(RegCap_optimization_variables_wind_list[ i ][ ts ])
                    reserve_terms.append(Rev_optimization_variables_wind_list[ i ][ ts ])

            if enable_dg and self.Vpp_temp_devices.get("DG"):
                for i in range(len(self.Vpp_temp_devices[ "DG" ])):
                    energy_terms.append(Energy_optimization_variables_dg_list[ i ][ ts ])
                    reg_cap_terms.append(RegCap_optimization_variables_dg_list[ i ][ ts ])
                    reserve_terms.append(Rev_optimization_variables_dg_list[ i ][ ts ])

            if enable_ess and self.Vpp_temp_devices.get("ESS"):
                for i in range(len(self.Vpp_temp_devices[ "ESS" ])):
                    energy_terms.append(Energy_optimization_variables_ess_discharge_list[ i ][ ts ] -
                                        Energy_optimization_variables_ess_charge_list[ i ][ ts ])
                    reg_cap_terms.append(RegCap_optimization_variables_ess_list[ i ][ ts ])
                    reserve_terms.append(Rev_optimization_variables_ess_list[ i ][ ts ])

            if enable_dl and self.Vpp_temp_devices.get("DL"):
                for i in range(len(self.Vpp_temp_devices[ "DL" ])):
                    energy_terms.append(-Energy_optimization_variables_dl_list[ i ][ ts ])
                    reg_cap_terms.append(RegCap_optimization_variables_dl_list[ i ][ ts ])
                    reserve_terms.append(Rev_optimization_variables_dl_list[ i ][ ts ])

            # --- 使用 mdl.sum() 对列表求和 ---
            vpp_actual_energy_at_ts = mdl.sum(energy_terms) - self.BusLoad[ ts ]
            vpp_actual_reg_cap_at_ts = mdl.sum(reg_cap_terms)
            vpp_actual_reserve_at_ts = mdl.sum(reserve_terms)

            # --- 建立偏差约束 ---
            mdl.add_constraint(
                vpp_actual_energy_at_ts - cleared_contracts.get('energy', [ 0 ] * time_points)[ ts ] == energy_pos_dev[
                    ts ] - energy_neg_dev[ ts ])
            mdl.add_constraint(vpp_actual_reg_cap_at_ts - cleared_contracts.get('reg_cap', [ 0 ] * time_points)[ ts ] ==
                               reg_cap_pos_dev[ ts ] - reg_cap_neg_dev[ ts ])
            mdl.add_constraint(
                vpp_actual_reserve_at_ts - cleared_contracts.get('reserve', [ 0 ] * time_points)[ ts ] == rev_pos_dev[
                    ts ] - rev_neg_dev[ ts ])

        # =================================================================
        # PART 5: RESTRUCTURED OBJECTIVE FUNCTION
        # =================================================================
        deviation_penalty = (mdl.sum(energy_pos_dev) + mdl.sum(energy_neg_dev)) * penalty_factor_energy + \
                            (mdl.sum(reg_cap_pos_dev) + mdl.sum(reg_cap_neg_dev) + mdl.sum(rev_pos_dev) + mdl.sum(
                                rev_neg_dev)) * penalty_factor_ancillary

        internal_operating_cost = mdl.sum([ ])
        time_interval = 0.25

        if enable_dg and self.Vpp_temp_devices.get("DG"):
            dg_cost = mdl.sum(
                (getattr(d, "Cost_energy_a", 0) + getattr(d, "Cost_energy_b", 0) *
                 Energy_optimization_variables_dg_list[ i ][ ts ] + getattr(d, "Cost_energy_c", 0) *
                 Energy_optimization_variables_dg_list[ i ][ ts ] ** 2 +
                 getattr(d, "Cost_Reg_Cap", 0) * RegCap_optimization_variables_dg_list[ i ][ ts ] +
                 getattr(d, "Cost_Rev", 0) * Rev_optimization_variables_dg_list[ i ][ ts ]) * time_interval
                for i, d in enumerate(self.Vpp_temp_devices[ "DG" ]) for ts in range(time_points)
            )
            internal_operating_cost += dg_cost

        if enable_ess and self.Vpp_temp_devices.get("ESS"):
            ess_cost = mdl.sum(
                (getattr(d, "Cost_energy", 0) * (Energy_optimization_variables_ess_charge_list[ i ][ ts ] +
                                                 Energy_optimization_variables_ess_discharge_list[ i ][ ts ]) +
                 getattr(d, "Cost_Reg_Cap", 0) * RegCap_optimization_variables_ess_list[ i ][ ts ] +
                 getattr(d, "Cost_Rev", 0) * Rev_optimization_variables_ess_list[ i ][ ts ]) * time_interval
                for i, d in enumerate(self.Vpp_temp_devices[ "ESS" ]) for ts in range(time_points)
            )
            internal_operating_cost += ess_cost

        DL_SHORTFALL_PENALTY_PER_KWH = 50.0  #

        dl_task_penalty = mdl.sum(dl_shortfall_vars) * DL_SHORTFALL_PENALTY_PER_KWH

        # mdl.minimize(deviation_penalty + internal_operating_cost)
        mdl.minimize(deviation_penalty+dl_task_penalty+internal_operating_cost)

        # =================================================================
        # PART 6: SOLVE & PROCESS RESULTS
        # =================================================================
        try:
            solution = mdl.solve()
            if solution:
                print(f"✅ 经济调度成功。")
                print(f"   - 目标函数值 (加权偏差+成本): {solution.get_objective_value():.4f}")
                print(f"   - 内部运行总成本: {internal_operating_cost.solution_value:.4f}")
                print(f"   - 总偏差惩罚: {deviation_penalty.solution_value:.4f}")

                # 构建详细的返回结果
                result = {
                    "status": "success",
                    "summary": {
                        "message": "Economic dispatch successful.",
                        "objective_value": solution.get_objective_value(),
                        "internal_cost": internal_operating_cost.solution_value,
                        "deviation_penalty": deviation_penalty.solution_value,
                        "owner_id": owner_id,
                        "time_points": time_points,
                        "start_ts": start_ts,
                        "mode": "economic_dispatch",
                    },
                    "vpp_data": {},
                    "device_data": {},
                }

                # 重新计算VPP层面的聚合出力
                vpp_energy_output = []
                vpp_regulation_capacity = []
                vpp_reserve = []
                for ts in range(time_points):
                    generation_sum, load_sum, reg_cap_sum, reserve_sum = 0, 0, 0, 0
                    if enable_dg and self.Vpp_temp_devices.get("DG"):
                        generation_sum += sum(
                            Energy_optimization_variables_dg_list[i][ts].solution_value for i in range(len(self.Vpp_temp_devices["DG"])))
                        reg_cap_sum += sum(
                            RegCap_optimization_variables_dg_list[i][ts].solution_value for i in range(len(self.Vpp_temp_devices["DG"])))
                        reserve_sum += sum(
                            Rev_optimization_variables_dg_list[i][ts].solution_value for i in range(len(self.Vpp_temp_devices["DG"])))
                    if enable_pv and self.Vpp_temp_devices.get("PV"):
                        generation_sum += sum(
                            Energy_optimization_variables_pv_list[i][ts].solution_value for i in range(len(self.Vpp_temp_devices["PV"])))
                        reg_cap_sum += sum(
                            RegCap_optimization_variables_pv_list[i][ts].solution_value for i in range(len(self.Vpp_temp_devices["PV"])))
                        reserve_sum += sum(
                            Rev_optimization_variables_pv_list[i][ts].solution_value for i in range(len(self.Vpp_temp_devices["PV"])))
                    if enable_wind and self.Vpp_temp_devices.get("WIND"):
                        generation_sum += sum(
                            Energy_optimization_variables_wind_list[i][ts].solution_value for i in
                            range(len(self.Vpp_temp_devices["WIND"])))
                        reg_cap_sum += sum(
                            RegCap_optimization_variables_wind_list[i][ts].solution_value for i in
                            range(len(self.Vpp_temp_devices["WIND"])))
                        reserve_sum += sum(
                            Rev_optimization_variables_wind_list[i][ts].solution_value for i in
                            range(len(self.Vpp_temp_devices["WIND"])))
                    if enable_ess and self.Vpp_temp_devices.get("ESS"):
                        generation_sum += sum(
                            Energy_optimization_variables_ess_discharge_list[i][ts].solution_value for i in
                            range(len(self.Vpp_temp_devices["ESS"])))
                        load_sum += sum(Energy_optimization_variables_ess_charge_list[i][ts].solution_value for i in
                                      range(len(self.Vpp_temp_devices["ESS"])))
                        reg_cap_sum += sum(
                            RegCap_optimization_variables_ess_list[i][ts].solution_value for i in
                            range(len(self.Vpp_temp_devices["ESS"])))
                        reserve_sum += sum(
                            Rev_optimization_variables_ess_list[i][ts].solution_value for i in
                            range(len(self.Vpp_temp_devices["ESS"])))
                    if enable_dl and self.Vpp_temp_devices.get("DL"):
                        load_sum += sum(
                            Energy_optimization_variables_dl_list[i][ts].solution_value for i in range(len(self.Vpp_temp_devices["DL"])))
                        reg_cap_sum += sum(
                            RegCap_optimization_variables_dl_list[i][ts].solution_value for i in range(len(self.Vpp_temp_devices["DL"])))
                        reserve_sum += sum(
                            Rev_optimization_variables_dl_list[i][ts].solution_value for i in range(len(self.Vpp_temp_devices["DL"])))

                    vpp_energy_output.append(generation_sum - load_sum - self.BusLoad[ts])
                    vpp_regulation_capacity.append(reg_cap_sum)
                    vpp_reserve.append(reserve_sum)

                result["vpp_data"] = {
                    "energy_output": vpp_energy_output,
                    "regulation_capacity": vpp_regulation_capacity,
                    "reserve": vpp_reserve
                }

                # 填充各设备类型的详细数据
                if enable_dg and self.Vpp_temp_devices.get("DG"):
                    result["device_data"]["dg"] = {}
                    for i, device in enumerate(self.Vpp_temp_devices["DG"]):
                        result["device_data"]["dg"][device.SimpleID] = {
                            "device_info": device.to_dict(),
                            "energy_output": [v.solution_value for v in Energy_optimization_variables_dg_list[i]],
                            "regulation_capacity": [v.solution_value for v in RegCap_optimization_variables_dg_list[i]],
                            "reserve": [v.solution_value for v in Rev_optimization_variables_dg_list[i]],
                        }
                if enable_pv and self.Vpp_temp_devices.get("PV"):
                    result["device_data"]["pv"] = {}
                    for i, device in enumerate(self.Vpp_temp_devices["PV"]):
                        result["device_data"]["pv"][device.SimpleID] = {
                            "device_info": device.to_dict(),
                            "energy_output": [v.solution_value for v in Energy_optimization_variables_pv_list[i]],
                            "regulation_capacity": [v.solution_value for v in RegCap_optimization_variables_pv_list[i]],
                            "reserve": [v.solution_value for v in Rev_optimization_variables_pv_list[i]],
                        }
                if enable_wind and self.Vpp_temp_devices.get("WIND"):
                    result["device_data"]["wind"] = {}
                    for i, device in enumerate(self.Vpp_temp_devices["WIND"]):
                        result["device_data"]["wind"][device.SimpleID] = {
                            "device_info": device.to_dict(),
                            "energy_output": [v.solution_value for v in Energy_optimization_variables_wind_list[i]],
                            "regulation_capacity": [v.solution_value for v in RegCap_optimization_variables_wind_list[i]],
                            "reserve": [v.solution_value for v in Rev_optimization_variables_wind_list[i]],
                        }
                if enable_ess and self.Vpp_temp_devices.get("ESS"):
                    result["device_data"]["ess"] = {}
                    for i, device in enumerate(self.Vpp_temp_devices["ESS"]):
                        result["device_data"]["ess"][device.SimpleID] = {
                            "device_info": device.to_dict(),
                            "charge_power": [v.solution_value for v in Energy_optimization_variables_ess_charge_list[i]],
                            "discharge_power": [v.solution_value for v in Energy_optimization_variables_ess_discharge_list[i]],
                            "soc_state": [v.solution_value for v in Energy_optimization_variables_soc_list[i]],
                            "regulation_capacity": [v.solution_value for v in RegCap_optimization_variables_ess_list[i]],
                            "reserve": [v.solution_value for v in Rev_optimization_variables_ess_list[i]],
                        }
                if enable_dl and self.Vpp_temp_devices.get("DL"):
                    result["device_data"]["dl"] = {}
                    for i, device in enumerate(self.Vpp_temp_devices["DL"]):
                        result["device_data"]["dl"][device.SimpleID] = {
                            "device_info": device.to_dict(),
                            "power_input": [v.solution_value for v in Energy_optimization_variables_dl_list[i]],
                            "capacity_state": [v.solution_value for v in Capacity_optimization_variables_dl_list[i]],
                            "regulation_capacity": [v.solution_value for v in RegCap_optimization_variables_dl_list[i]],
                            "reserve": [v.solution_value for v in Rev_optimization_variables_dl_list[i]],
                        }

                return result

            else:
                print("❌ 经济调度失败: 模型不可行。启动建模诊断分析...")
                cr = ConflictRefiner()
                conflicts = cr.refine_conflict(mdl)
                if conflicts:
                    print("发现以下核心冲突约束:")
                    conflicts.display()
                    return {"status": "infeasible", "message": "Model is infeasible. Core conflicts identified.",
                            "conflicts": [ str(c.element) for c in conflicts ]}
                else:
                    print("冲突精炼器未找到明确的冲突。")
                    return {"status": "infeasible",
                            "message": "Model is infeasible, but ConflictRefiner found no specific conflicts."}

        except Exception as e:
            print(f"❌ 经济调度求解器错误: {e}")
            return {"status": "error", "message": str(e)}
