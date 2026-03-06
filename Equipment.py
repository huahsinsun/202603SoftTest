"""
该模块定义了虚拟电厂(VPP)的核心设备模型，包括基础设备抽象类Equipment及其具体实现类。

Equipment类是所有VPP设备的基类，定义了设备的基本属性和方法。
具体设备类型(如PV光伏、ESS储能等)继承自Equipment类并实现特定功能。

主要功能:
- 定义设备基础属性(设备ID、类型、状态等)
- 提供设备功率输出范围限制
- 记录设备成本参数(能量成本、调频成本等)
- 管理设备分组控制
- 记录设备安装和更新时间

涉及到成本系数的,输出功率的设备尺度信息均使用KW/KWH尺度计算,参数录入时需要尤其注意

注意
pv1 = PV(DeviceID="PV001", OwnerID="USER123")  # SimpleID 自动生成为 "USER123-PV-001"
pv2 = PV(DeviceID="PV002", OwnerID="USER123")  # SimpleID 自动生成为 "USER123-PV-002"
pv3 = PV(DeviceID="PV003", OwnerID="USER456")  # SimpleID 自动生成为 "USER456-PV-001"
"""

from abc import ABC, abstractmethod
from datetime import time, datetime


class Equipment(ABC):
    _simple_id_counts = {}

    def __init__(
        self,
        DeviceID,  # 设备唯一标识
        OwnerID,  # 设备所属用户或户号
        DeviceType,  # 设备类型 (如 "PV", "Wind", "DG", "ESS", "DL", "TCR")
        Status,  # 设备状态 (如 1:正常, 0:停用)
        Pout_min,
        Pout_max,
        Cost_energy,  # 单位成本/成本系数
        Cost_Reg_Cap,  # 调频容量成本
        Cost_Rev,  # 备用市场成本
        SimpleID=None,  # 简化设备ID (OwnerID-DeviceType-序号)
        ControlGroupID=None,  # 控制组标识 (可选，用于控制分组)
        InstalledDate=None,  # 安装日期
        LastUpdateTime=None,  # 最后更新时间
        # 投标比例限制因子 - 类似MATLAB中的硬编码参数
        *,
        bid_ratio_energy=None,  # 电能量市场投标比例因子 (0.0-1.0)
        bid_ratio_reg_cap=None,  # 调频容量市场投标比例因子 (0.0-1.0)
        bid_ratio_rev=None,  # 备用市场投标比例因子 (0.0-1.0)
    ):
        self.DeviceID = DeviceID
        self.OwnerID = OwnerID
        self.DeviceType = DeviceType
        self.Status = Status
        self.ControlGroupID = ControlGroupID
        self.Pout_min = Pout_min
        self.Pout_max = Pout_max
        self.Cost_energy = Cost_energy
        self.Cost_Reg_Cap = Cost_Reg_Cap
        self.Cost_Rev = Cost_Rev
        self.InstalledDate = InstalledDate
        self.LastUpdateTime = LastUpdateTime

        # 投标比例限制因子 - 如果未设置则使用默认值
        self.bid_ratio_energy = (
            bid_ratio_energy
            if bid_ratio_energy is not None
            else self._get_default_bid_ratio_energy()
        )
        self.bid_ratio_reg_cap = (
            bid_ratio_reg_cap
            if bid_ratio_reg_cap is not None
            else self._get_default_bid_ratio_reg_cap()
        )
        self.bid_ratio_rev = (
            bid_ratio_rev
            if bid_ratio_rev is not None
            else self._get_default_bid_ratio_rev()
        )

        # Generate SimpleID if not provided
        if SimpleID is None:
            key = (self.OwnerID, self.DeviceType)
            current_count = Equipment._simple_id_counts.get(key, 0)
            current_count += 1
            self.SimpleID = f"{self.OwnerID}-{self.DeviceType}-{current_count:03d}"
            Equipment._simple_id_counts[key] = current_count
        else:
            self.SimpleID = SimpleID

    def _get_default_bid_ratio_energy(self):
        """获取设备类型默认的电能量投标比例因子"""
        defaults = {"PV": 1.0, "WIND": 1.0, "DG": 1, "ESS": 1.0, "DL": 1.0, "TCR": 1.0}
        return defaults.get(self.DeviceType, 0.8)

    def _get_default_bid_ratio_reg_cap(self):
        """获取设备类型默认的调频容量投标比例因子"""
        defaults = {
            "PV": 0.15,
            "WIND": 0.15,
            "DG": 0.15,
            "ESS": 0.3,
            "DL": 0.1,
            "TCR": 0.05,
        }
        return defaults.get(self.DeviceType, 0.15)

    def _get_default_bid_ratio_rev(self):
        """获取设备类型默认的备用投标比例因子"""
        defaults = {
            "PV": 0.1,  # 不允许光伏参与备用
            "WIND": 0.1,  # 不允许风电参与备用
            "DG": 0.2,
            "ESS": 0.3,
            "DL": 0.15,
            "TCR": 0.1,
        }
        return defaults.get(self.DeviceType, 0.2)

    def get_bid_limit_energy_max(self):
        """获取电能量投标上限"""
        return self.bid_ratio_energy * self.Pout_max

    def get_bid_limit_reg_cap_max(self):
        """获取调频容量投标上限"""
        return self.bid_ratio_reg_cap * self.Pout_max

    def get_bid_limit_rev_max(self):
        """获取备用投标上限"""
        return self.bid_ratio_rev * self.Pout_max

    def update_bid_ratios(self, energy=None, reg_cap=None, rev=None):
        """批量更新投标比例因子"""
        if energy is not None:
            self.bid_ratio_energy = energy
        if reg_cap is not None:
            self.bid_ratio_reg_cap = reg_cap
        if rev is not None:
            self.bid_ratio_rev = rev

    def __str__(self):
        """返回设备的可读字符串表示"""
        return f"{self.DeviceType}({self.SimpleID}): Status={self.Status}, Pout=[{self.Pout_min}, {self.Pout_max}], Cost={self.Cost_energy}"

    def __repr__(self):
        """返回设备的详细字符串表示"""
        return self.__str__()

    @abstractmethod
    def to_dict(self):
        pass


class PV(Equipment):
    def __init__(
        self,
        DeviceID,
        OwnerID,
        Status=0,
        Pout_min=0.0,
        Pout_max=0.0,
        Cost_energy=0.0,
        Cost_Reg_Cap=0.0,
        Cost_Rev=0.0,
        ControlGroupID=None,
        InstalledDate=None,
        LastUpdateTime=None,
    ):
        super().__init__(
            DeviceID,
            OwnerID,
            "PV",
            Status,
            Pout_min,
            Pout_max,
            Cost_energy,
            Cost_Reg_Cap,
            Cost_Rev,
            ControlGroupID=ControlGroupID,
            InstalledDate=InstalledDate,
            LastUpdateTime=LastUpdateTime,
        )

    def __str__(self):
        return (
            f"PV(DeviceID={self.DeviceID}, OwnerID={self.OwnerID}, "
            f"Status={self.Status}, Pout_min={self.Pout_min}, "
            f"Pout_max={self.Pout_max}, Cost_energy={self.Cost_energy}, "
            f"Cost_Reg_Cap={self.Cost_Reg_Cap}, "
            f"Cost_Rev={self.Cost_Rev}, ControlGroupID={self.ControlGroupID}, "
            f"InstalledDate={self.InstalledDate}, LastUpdateTime={self.LastUpdateTime})"
        )

    def to_dict(self):
        return {
            "DeviceID": self.DeviceID,
            "OwnerID": self.OwnerID,
            "DeviceType": self.DeviceType,
            "Status": self.Status,
            "Pout_min": self.Pout_min,
            "Pout_max": self.Pout_max,
            "Cost_energy": self.Cost_energy,
            "Cost_Reg_Cap": self.Cost_Reg_Cap,
            "Cost_Rev": self.Cost_Rev,
            "ControlGroupID": self.ControlGroupID,
            "InstalledDate": self.InstalledDate,
            "LastUpdateTime": self.LastUpdateTime,
        }

    def get_forecast_power(self):
        """
        获取光伏出力预测。
        优先使用由API层在运行时注入的、针对当前优化窗口的精确预测数据。
        如果不存在，则回退到从全局文件中读取。
        """
        if hasattr(self, "window_forecast") and self.window_forecast is not None:
            # print(f"DEBUG: PV {self.DeviceID} 使用了注入的窗口预测数据。")
            return self.window_forecast
        # Fallback to the old global method
        # print(f"WARN: PV {self.DeviceID} 回退到全局预测数据加载方法。")
        if __package__:
            from .ExternalTools import get_RES_forecast_data
        else:
            from ExternalTools import get_RES_forecast_data

        return get_RES_forecast_data(self.DeviceID, self.DeviceType)


class WIND(Equipment):
    def __init__(
        self,
        DeviceID,
        OwnerID,
        Status=0,
        Pout_min=0.0,
        Pout_max=0.0,
        Cost_energy=0.0,
        Cost_Reg_Cap=0.0,
        Cost_Rev=0.0,
        ControlGroupID=None,
        InstalledDate=None,
        LastUpdateTime=None,
    ):
        super().__init__(
            DeviceID,
            OwnerID,
            "WIND",
            Status,
            Pout_min,
            Pout_max,
            Cost_energy,
            Cost_Reg_Cap,
            Cost_Rev,
            ControlGroupID=ControlGroupID,
            InstalledDate=InstalledDate,
            LastUpdateTime=LastUpdateTime,
        )

    def __str__(self):
        return (
            f"WIND(DeviceID={self.DeviceID}, OwnerID={self.OwnerID}, "
            f"Status={self.Status}, Pout_min={self.Pout_min}, "
            f"Pout_max={self.Pout_max}, Cost_energy={self.Cost_energy}, "
            f"Cost_Reg_Cap={self.Cost_Reg_Cap}, "
            f"Cost_Rev={self.Cost_Rev}, ControlGroupID={self.ControlGroupID}, "
            f"InstalledDate={self.InstalledDate}, LastUpdateTime={self.LastUpdateTime})"
        )

    def to_dict(self):
        return {
            "DeviceID": self.DeviceID,
            "OwnerID": self.OwnerID,
            "DeviceType": self.DeviceType,
            "Status": self.Status,
            "Pout_min": self.Pout_min,
            "Pout_max": self.Pout_max,
            "Cost_energy": self.Cost_energy,
            "Cost_Reg_Cap": self.Cost_Reg_Cap,
            "Cost_Rev": self.Cost_Rev,
            "ControlGroupID": self.ControlGroupID,
            "InstalledDate": self.InstalledDate,
            "LastUpdateTime": self.LastUpdateTime,
        }

    def get_forecast_power(self):
        """
        获取风电出力预测。
        优先使用注入的窗口预测数据。
        """
        if hasattr(self, "window_forecast") and self.window_forecast is not None:
            # print(f"DEBUG: WIND {self.DeviceID} 使用了注入的窗口预测数据。")
            return self.window_forecast

        # Fallback to the old global method
        # print(f"WARN: WIND {self.DeviceID} 回退到全局预测数据加载方法。")
        if __package__:
            from .ExternalTools import get_RES_forecast_data
        else:
            from ExternalTools import get_RES_forecast_data

        return get_RES_forecast_data(self.DeviceID, self.DeviceType)


class DG(Equipment):
    def __init__(
        self,
        DeviceID,
        OwnerID,
        Status=0,
        Pout_min=0,
        Pout_max=0,
        Cost_Reg_Cap=0,
        Cost_Rev=0,
        ControlGroupID=None,
        InstalledDate=None,
        LastUpdateTime=None,
        Cost_energy_a=0,
        Cost_energy_b=0,
        Cost_energy_c=0,
    ):
        super().__init__(
            DeviceID,
            OwnerID,
            "DG",
            Status,
            Pout_min,
            Pout_max,
            # DG has different energy costs, so pass 0 for the base class energy cost
            0,
            Cost_Reg_Cap,
            Cost_Rev,
            ControlGroupID=ControlGroupID,
            InstalledDate=InstalledDate,
            LastUpdateTime=LastUpdateTime,
        )
        self.Cost_energy_a = Cost_energy_a
        self.Cost_energy_b = Cost_energy_b
        self.Cost_energy_c = Cost_energy_c

    def __str__(self):
        return (
            f"DG(DeviceID={self.DeviceID}, OwnerID={self.OwnerID}, "
            f"Pout_max={self.Pout_max}, "
            f"Cost_coeffs(a={self.Cost_energy_a}, b={self.Cost_energy_b}, c={self.Cost_energy_c}), "
            f"Cost_Reg_Cap={self.Cost_Reg_Cap}, "
            f"Cost_Rev={self.Cost_Rev})"
        )

    def to_dict(self):
        return {
            "DeviceID": self.DeviceID,
            "OwnerID": self.OwnerID,
            "DeviceType": self.DeviceType,
            "Status": self.Status,
            "Pout_min": self.Pout_min,
            "Pout_max": self.Pout_max,
            "Cost_energy_a": self.Cost_energy_a,
            "Cost_energy_b": self.Cost_energy_b,
            "Cost_energy_c": self.Cost_energy_c,
            "Cost_Reg_Cap": self.Cost_Reg_Cap,
            "Cost_Rev": self.Cost_Rev,
            "ControlGroupID": self.ControlGroupID,
            "InstalledDate": self.InstalledDate,
            "LastUpdateTime": self.LastUpdateTime,
        }


class ESS(Equipment):
    def __init__(
        self,
        DeviceID,
        OwnerID,
        Status=0,
        Capacity=0.0,
        Pin_min=-0.0,
        Pin_max=0.0,
        Pout_min=0.0,
        Pout_max=0.0,
        SelfDischargeRate=0.98,
        TransferEfficiency=0.95,
        InitialSOC=0.0,
        CurrentSOC=None,
        Cost_energy=0.0,
        Cost_Reg_Cap=0.0,
        Cost_Rev=0.0,
        ControlGroupID=None,
        InstalledDate=None,
        LastUpdateTime=None,
    ):
        super().__init__(
            DeviceID,
            OwnerID,
            "ESS",
            Status,
            Pout_min,
            Pout_max,
            Cost_energy,
            Cost_Reg_Cap,
            Cost_Rev,
            ControlGroupID=ControlGroupID,
            InstalledDate=InstalledDate,
            LastUpdateTime=LastUpdateTime,
        )
        self.Capacity = Capacity
        self.Pin_min = Pin_min
        self.Pin_max = Pin_max
        self.SelfDischargeRate = SelfDischargeRate
        self.TransferEfficiency = TransferEfficiency
        self.InitialSOC = InitialSOC
        self.CurrentSOC = CurrentSOC if CurrentSOC is not None else InitialSOC
        # The Cost_energy is already passed to the super class, this line is redundant
        # self.Cost_energy = Cost_energy

    def get_ess_charge_max(self, bid_ratio_charge=1):
        """获取储能充电上限"""
        return bid_ratio_charge * self.Pin_max

    def get_ess_discharge_max(self):
        return self.Pout_max

    def __str__(self):
        return (
            f"ESS(DeviceID={self.DeviceID}, OwnerID={self.OwnerID}, "
            f"Capacity={self.Capacity}, Pout_max={self.Pout_max}, "
            f"Pin_max={self.Pin_max}, InitialSOC={self.InitialSOC}, "
            f"Cost_Reg_Cap={self.Cost_Reg_Cap}, "
            f"Cost_Rev={self.Cost_Rev})"
        )

    def to_dict(self):
        return {
            "DeviceID": self.DeviceID,
            "OwnerID": self.OwnerID,
            "DeviceType": self.DeviceType,
            "Status": self.Status,
            "Capacity": self.Capacity,
            "Pin_min": self.Pin_min,
            "Pin_max": self.Pin_max,
            "Pout_min": self.Pout_min,
            "Pout_max": self.Pout_max,
            "SelfDischargeRate": self.SelfDischargeRate,
            "TransferEfficiency": self.TransferEfficiency,
            "InitialSOC": self.InitialSOC,
            "CurrentSOC": self.CurrentSOC,
            "Cost_energy": self.Cost_energy,
            "Cost_Reg_Cap": self.Cost_Reg_Cap,
            "Cost_Rev": self.Cost_Rev,
            "ControlGroupID": self.ControlGroupID,
            "InstalledDate": self.InstalledDate,
            "LastUpdateTime": self.LastUpdateTime,
        }


class DL(Equipment):
    def __init__(
        self,
        DeviceID,
        OwnerID,
        Status=0,
        Cap_min=0.0,
        Cap_max=0.0,
        Pin_min=-0.0,
        Pin_max=0.0,
        SelfDischargeRate=0.98,
        TransferEfficiency=0.95,
        InitialCapacity=0.0, # 是一个百分比
        cumulative_capacity=None, # 是累计容量，不是最大容量
        StartTime=None,
        EndTime=None,
        RequiredDemand=0.0,
        Cost_energy=0.0,
        Cost_Reg_Cap=0.0,
        Cost_Rev=0.0,
        ControlGroupID=None,
        InstalledDate=None,
        LastUpdateTime=None,
    ):
        super().__init__(
            DeviceID,
            OwnerID,
            "DL",
            Status,
            0.0,  # Pout_min设为0，因为只能充电
            0.0,  # Pout_max设为0，因为只能充电
            Cost_energy,
            Cost_Reg_Cap,
            Cost_Rev,
            ControlGroupID=ControlGroupID,
            InstalledDate=InstalledDate,
            LastUpdateTime=LastUpdateTime,
        )
        self.Cap_min = Cap_min
        self.Cap_max = Cap_max
        self.Pin_min = Pin_min
        self.Pin_max = Pin_max
        self.SelfDischargeRate = SelfDischargeRate
        self.TransferEfficiency = TransferEfficiency
        self.InitialCapacity = InitialCapacity
        if isinstance(StartTime, str):
            self.StartTime = datetime.strptime(StartTime, "%H:%M").time()
        else:
            self.StartTime = StartTime or time(0, 0)

        if isinstance(EndTime, str):
            self.EndTime = datetime.strptime(EndTime, "%H:%M").time()
        else:
            self.EndTime = EndTime or time(23, 45)
        self.RequiredDemand = RequiredDemand
        # ==================== 【核心修正】 ====================
        # 2. 在对象创建时，就将百分比的InitialCapacity转换为绝对值的cumulative_capacity
        
        # 首先计算出初始的绝对容量值
        initial_absolute_capacity = self.InitialCapacity * self.Cap_max
        
        # self.cumulative_capacity 要么是外部传入的、更新后的绝对值，
        # 要么是从初始百分比计算出的初始绝对值。
        self.cumulative_capacity = (
            cumulative_capacity if cumulative_capacity is not None else initial_absolute_capacity
        )
        
        
        
        
        self.cumulative_capacity = (
            cumulative_capacity if cumulative_capacity is not None else InitialCapacity
        )
        # The Cost_energy is already passed to the super class, this line is redundant
        # self.Cost_energy = Cost_energy

    def get_bid_limit_energy_max(self):
        """获取电能量投标上限 - DL设备基于Pin_max"""
        return self.bid_ratio_energy * self.Pin_max

    def get_bid_limit_reg_cap_max(self):
        """获取调频容量投标上限 - DL设备基于Pin_max"""
        return self.bid_ratio_reg_cap * self.Pin_max

    def get_bid_limit_rev_max(self):
        """获取备用投标上限 - DL设备基于Pin_max"""
        return self.bid_ratio_rev * self.Pin_max

    def __str__(self):
        return (
            f"DL(DeviceID={self.DeviceID}, OwnerID={self.OwnerID}, "
            f"Cap_max={self.Cap_max}, Pin_max={self.Pin_max}, "
            f"RequiredDemand={self.RequiredDemand}, "
            f"Cost_Reg_Cap={self.Cost_Reg_Cap}, "
            f"Cost_Rev={self.Cost_Rev})"
        )

    def to_dict(self):
        return {
            "DeviceID": self.DeviceID,
            "OwnerID": self.OwnerID,
            "DeviceType": self.DeviceType,
            "Status": self.Status,
            "Cap_min": self.Cap_min,
            "Cap_max": self.Cap_max,
            "Pin_min": self.Pin_min,
            "Pin_max": self.Pin_max,
            "SelfDischargeRate": self.SelfDischargeRate,
            "TransferEfficiency": self.TransferEfficiency,
            "InitialCapacity": self.InitialCapacity,
            "StartTime": self.StartTime.strftime("%H:%M") if self.StartTime else None,
            "EndTime": self.EndTime.strftime("%H:%M") if self.EndTime else None,
            "RequiredDemand": self.RequiredDemand,
            "Cost_energy": self.Cost_energy,
            "Cost_Reg_Cap": self.Cost_Reg_Cap,
            "Cost_Rev": self.Cost_Rev,
            "ControlGroupID": self.ControlGroupID,
            "InstalledDate": self.InstalledDate,
            "LastUpdateTime": self.LastUpdateTime,
        }


class TCR(Equipment):
    def __init__(
        self,
        DeviceID,
        OwnerID,
        Status=0,
        Temp_min=0.0,
        Temp_max=0.0,
        Pin_min=-0.0,
        Pin_max=0.0,
        TempDecayRate=0.9,
        ThermalEfficiency=0.7,
        InitialTemp=0.0,
        Cost_energy=0.0,
        Cost_Reg_Cap=0.0,
        Cost_Rev=0.0,
        ControlGroupID=None,
        InstalledDate=None,
        LastUpdateTime=None,
    ):
        super().__init__(
            DeviceID,
            OwnerID,
            "TCR",
            Status,
            0.0,  # Pout_min设为0，因为只能充电
            0.0,  # Pout_max设为0，因为只能充电
            Cost_energy,
            Cost_Reg_Cap,
            Cost_Rev,
            ControlGroupID=ControlGroupID,
            InstalledDate=InstalledDate,
            LastUpdateTime=LastUpdateTime,
        )
        self.Temp_min = Temp_min
        self.Temp_max = Temp_max
        self.Pin_min = Pin_min
        self.Pin_max = Pin_max
        self.TempDecayRate = TempDecayRate
        self.ThermalEfficiency = ThermalEfficiency
        self.InitialTemp = InitialTemp
        # The Cost_energy is already passed to the super class, this line is redundant
        # self.Cost_energy = Cost_energy

    def get_bid_limit_energy_max(self):
        """获取电能量投标上限 - TCR设备基于Pin_max"""
        return self.bid_ratio_energy * self.Pin_max

    def get_bid_limit_reg_cap_max(self):
        """获取调频容量投标上限 - TCR设备基于Pin_max"""
        return self.bid_ratio_reg_cap * self.Pin_max

    def get_bid_limit_rev_max(self):
        """获取备用投标上限 - TCR设备基于Pin_max"""
        return self.bid_ratio_rev * self.Pin_max

    def __str__(self):
        return (
            f"TCR(DeviceID={self.DeviceID}, OwnerID={self.OwnerID}, "
            f"Temp_max={self.Temp_max}, Pin_max={self.Pin_max}, "
            f"InitialTemp={self.InitialTemp}, "
            f"Cost_Reg_Cap={self.Cost_Reg_Cap}, "
            f"Cost_Rev={self.Cost_Rev})"
        )

    def to_dict(self):
        return {
            "DeviceID": self.DeviceID,
            "OwnerID": self.OwnerID,
            "DeviceType": self.DeviceType,
            "Status": self.Status,
            "Temp_min": self.Temp_min,
            "Temp_max": self.Temp_max,
            "Pin_min": self.Pin_min,
            "Pin_max": self.Pin_max,
            "TempDecayRate": self.TempDecayRate,
            "ThermalEfficiency": self.ThermalEfficiency,
            "InitialTemp": self.InitialTemp,
            "Cost_energy": self.Cost_energy,
            "Cost_Reg_Cap": self.Cost_Reg_Cap,
            "Cost_Rev": self.Cost_Rev,
            "ControlGroupID": self.ControlGroupID,
            "InstalledDate": self.InstalledDate,
            "LastUpdateTime": self.LastUpdateTime,
        }
