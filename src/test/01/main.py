import os

from ...my_draw import Plotter
from ...my_draw import Parser


PORT = os.getenv("PLOTTER_PORT")
if PORT is None:
    raise Exception("Please proveide the serial port of the plotter")

parser = Parser.from_file("src/test/01/a5.svg")
parser.interpolate()
parser.optimize_curves()
parser.scale_to_fit("a6")

plotter = Plotter(PORT)
plotter.init_gcode()
plotter.convert_curves(parser.curves)
plotter.finish_gcode()
plotter.exec_commands()
