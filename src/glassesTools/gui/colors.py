from imgui_bundle import imgui

error       = imgui.ImColor.hsv(0.9667,.88,.64).value
error_bright= imgui.ImColor.hsv(0.9667,.88,.93).value
error_dark  = imgui.ImColor.hsv(0.9667,.88,.43).value

warning         = imgui.ImColor.hsv(45/360,.97,1.).value
warning_bright  = imgui.ImColor.hsv(54/360,.97,1.).value
warning_dark    = imgui.ImColor.hsv(45/360,.97,.8).value

ok          = imgui.ImVec4(0.0000, 0.8500, 0.0000, 1.)

gray        = imgui.ImVec4(0.5000, 0.5000, 0.5000, 1.)