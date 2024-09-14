import os

from ..my_draw import Plotter
from ..my_draw import Parser


PORT = os.getenv("PLOTTER_PORT")
if PORT is None:
    raise Exception("Please proveide the serial port of the plotter")

TEST_SVG = os.getenv("PLOTTER_TEST_SVG")
if TEST_SVG is None:
    raise Exception("Please proveide a testing SVG")

parser = Parser.from_file(TEST_SVG)
parser.optimize_curves()

plotter = Plotter(PORT)
parser.scale_to_fit((135, 200))
plotter.convert_curves(parser.curves)
plotter.exec_commands()
