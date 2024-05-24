import enum

from . import utils

class EyeTracker(utils.AutoName):
    AdHawk_MindLink = enum.auto()
    Pupil_Core      = enum.auto()
    Pupil_Invisible = enum.auto()
    Pupil_Neon      = enum.auto()
    SMI_ETG         = enum.auto()
    SeeTrue_STONE   = enum.auto()
    Tobii_Glasses_2 = enum.auto()
    Tobii_Glasses_3 = enum.auto()
    Unknown         = enum.auto()
eye_tracker_names = [x.value for x in EyeTracker if x!=EyeTracker.Unknown]

EyeTracker.AdHawk_MindLink.color = utils.hex_to_rgba_0_1("#001D7A")
EyeTracker.Pupil_Core     .color = utils.hex_to_rgba_0_1("#E6194B")
EyeTracker.Pupil_Invisible.color = utils.hex_to_rgba_0_1("#3CB44B")
EyeTracker.Pupil_Neon     .color = utils.hex_to_rgba_0_1("#C6B41E")
EyeTracker.SMI_ETG        .color = utils.hex_to_rgba_0_1("#4363D8")
EyeTracker.SeeTrue_STONE  .color = utils.hex_to_rgba_0_1("#911EB4")
EyeTracker.Tobii_Glasses_2.color = utils.hex_to_rgba_0_1("#F58231")
EyeTracker.Tobii_Glasses_3.color = utils.hex_to_rgba_0_1("#F032E6")
EyeTracker.Unknown        .color = utils.hex_to_rgba_0_1("#393939")

def string_to_enum(device: str) -> EyeTracker:
    if isinstance(device, EyeTracker):
        return device

    if isinstance(device, str):
        if hasattr(EyeTracker, device):
            return getattr(EyeTracker, device)
        elif device in [e.value for e in EyeTracker]:
            return EyeTracker(device)
        else:
            raise ValueError(f"The string '{device}' is not a known eye tracker type. Known types: {[e.value for e in EyeTracker]}")
    else:
        raise ValueError(f"The variable 'device' should be a string with one of the following values: {[e.value for e in EyeTracker]}")


utils.register_type(utils.CustomTypeEntry(EyeTracker,'__enum.EyeTracker__',str, lambda x: getattr(EyeTracker, x.split('.')[1])))