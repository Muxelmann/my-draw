from ..plotter import Plotter
from ..plotter import Parser


parser = Parser.from_file("src/test/bezier.svg")

plotter = Plotter()
plotter.convert_curves(parser.curves)
plotter.exec_commands()
