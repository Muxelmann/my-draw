import os
import json

from ...my_draw import Plotter
from ...my_draw import Parser


PORT = os.getenv("PLOTTER_PORT")
if PORT is None:
    raise Exception("Please proveide the serial port of the plotter")

pen_color = "black"  # Or a color like rgb(136,136,136)

parser = Parser.from_file("src/test/02/stroke-and-fill.svg")
parser.scale_to_fit("a5")

print(json.dumps(parser.curves_for_filling, indent=" "))

plotter = Plotter(PORT)
plotter.init_gcode()

curves_to_stroke = parser.curves_to_stroke
if pen_color in curves_to_stroke.keys():
    curve = curves_to_stroke[pen_color]
    curve = parser.optimize(curve)
    curve = parser.interpolate(curve)
    plotter.convert_curves(curve)

curves_for_filling = parser.curves_for_filling
if pen_color in curves_for_filling.keys():
    curve = curves_for_filling[pen_color]
    curve = parser.optimize(curve)
    plotter.convert_curves(curve)

plotter.finish_gcode()

plotter.exec_commands()
