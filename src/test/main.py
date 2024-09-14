import os

from ..my_draw import Plotter
from ..my_draw import Parser


PORT = os.getenv("PLOTTER_PORT")
if PORT is None:
    raise Exception("Please proveide the serial port of the plotter")

parser = Parser.from_file("src/test/bezier.svg")

plotter = Plotter(PORT)
plotter.convert_curves(parser.curves)
plotter.exec_commands()
