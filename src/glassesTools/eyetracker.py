import enum

from . import json, utils

class EyeTracker(utils.AutoName):
    AdHawk_MindLink = enum.auto()
    Generic         = enum.auto()
    Meta_Aria_Gen_1 = enum.auto()
    Pupil_Core      = enum.auto()
    Pupil_Invisible = enum.auto()
    Pupil_Neon      = enum.auto()
    SeeTrue_STONE   = enum.auto()
    SMI_ETG         = enum.auto()
    Tobii_Glasses_2 = enum.auto()
    Tobii_Glasses_3 = enum.auto()
    VPS_19          = enum.auto()
    Unknown         = enum.auto()
eye_tracker_names = [x.value for x in EyeTracker if x!=EyeTracker.Unknown]

EyeTracker.AdHawk_MindLink.color = utils.hex_to_rgba_0_1("#F0A3FF")
EyeTracker.Generic        .color = utils.hex_to_rgba_0_1("#393939")
EyeTracker.Meta_Aria_Gen_1.color = utils.hex_to_rgba_0_1("#9DCC00")
EyeTracker.Pupil_Core     .color = utils.hex_to_rgba_0_1("#0075DC")
EyeTracker.Pupil_Invisible.color = utils.hex_to_rgba_0_1("#993F00")
EyeTracker.Pupil_Neon     .color = utils.hex_to_rgba_0_1("#4C005C")
EyeTracker.SeeTrue_STONE  .color = utils.hex_to_rgba_0_1("#005C31")
EyeTracker.SMI_ETG        .color = utils.hex_to_rgba_0_1("#2BCE48")
EyeTracker.Tobii_Glasses_2.color = utils.hex_to_rgba_0_1("#FFCC99")
EyeTracker.Tobii_Glasses_3.color = utils.hex_to_rgba_0_1("#94FFB5")
EyeTracker.VPS_19         .color = utils.hex_to_rgba_0_1("#8F7C00")
EyeTracker.Unknown        .color = utils.hex_to_rgba_0_1("#393939")
# other colors left over:
# #C20088, #003380, #FFA405, #FFA8BB, #426600, #FF0010, #5EF1F2, #00998F, #E0FF66, #740AFF, #990000, #FFFF80, #FFE100, #FF5005
# colors taken from P. Green-Armytage (2010). "A Colour Alphabet and the Limits of Colour Coding". Colour: Design & Creativity 5 (10): 1–23.


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


json.register_type(json.TypeEntry(EyeTracker,'__enum.EyeTracker__', utils.enum_val_2_str, lambda x: getattr(EyeTracker, x.split('.')[1])))