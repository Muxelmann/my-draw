import os

from ...my_draw import Plotter
from ...my_draw import Parser


PORT = os.getenv("PLOTTER_PORT")
if PORT is None:
    raise Exception("Please proveide the serial port of the plotter")

parser = Parser.from_file("src/test/01/a5.svg")
parser.scale_to_fit("a6")

curves = parser.curves
curves = parser.interpolate(curves)
curves = parser.optimize(curves)

plotter = Plotter(PORT)

plotter.init_gcode()
plotter.convert_curves(curves)
plotter.finish_gcode()

plotter.exec_commands()
