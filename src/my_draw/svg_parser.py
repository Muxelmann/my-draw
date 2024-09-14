from xml.etree import ElementTree
import re
import copy
import math

NAMESPACE = "{http://www.w3.org/2000/svg}"


class Parser:
    def __init__(
        self,
        svg_string: str,
        parse_on_init: bool = True,
        interpolate_on_init: bool = True,
        scale_format: str | None = None,
    ) -> None:
        self.curves = []

        self.current_curve = []
        self.current_transforms = []
        self.last_command = None

        root = ElementTree.fromstring(svg_string)
        self.viewbox = [float(c) for c in root.get("viewBox").split(" ")]

        if parse_on_init:
            self.parse(root)

            if interpolate_on_init:
                self.interpolate()

            if scale_format:
                self.scale_to_fit(scale_format)

    @staticmethod
    def from_file(path: str) -> "Parser":
        with open(path, "r") as f:
            return Parser(f.read())

    def parse(self, root: ElementTree.Element, transform: str | None = None) -> None:

        self.current_transforms.append(transform)

        for element in list(root):

            if element.tag == f"{NAMESPACE}path":
                self.convert_path(element.get("d"), transform)

            elif element.tag == f"{NAMESPACE}g":
                self.parse(element, element.get("transform"))

            elif element.tag == f"{NAMESPACE}rect":
                self.convert_rect(
                    float(element.get("x")),
                    float(element.get("y")),
                    float(element.get("width")),
                    float(element.get("height")),
                )

            elif element.tag == f"{NAMESPACE}circle":
                self.convert_circle(
                    float(element.get("cx")),
                    float(element.get("cy")),
                    float(element.get("r")),
                )

            elif element.tag == f"{NAMESPACE}defs":
                self.convert_defs(element)

            else:
                raise Exception(f"Unknown tag {element.tag}")

        self.current_transforms.pop()

    def tx_matrix(self, p: list) -> None:
        pass
        for i in range(len(self.current_curve)):
            self.current_curve[i] = [
                self.current_curve[i][0] * p[0]
                + self.current_curve[i][1] * p[2]
                + p[4],
                self.current_curve[i][0] * p[1]
                + self.current_curve[i][1] * p[3]
                + p[5],
            ]

    def tx_scale(self, p: list) -> None:
        if len(p) == 1:
            p = [p[0], p[0]]

        for i in range(len(self.current_curve)):
            self.current_curve[i] = [
                self.current_curve[i][0] * p[0],
                self.current_curve[i][1] * p[1],
            ]

    def tx_translate(self, p: list) -> None:
        if len(p) == 1:
            p = [p[0], 0]

        for i in range(len(self.current_curve)):
            self.current_curve[i] = [
                self.current_curve[i][0] + p[0],
                self.current_curve[i][1] + p[1],
            ]

    def apply_transforms(self) -> None:
        transform_list = [t for t in self.current_transforms if t is not None]

        if len(transform_list) == 0:
            return

        transforms = {
            "matrix": self.tx_matrix,
            "scale": self.tx_scale,
            "translate": self.tx_translate,
            # "rotate": None,
            # "skewX": None,
            # "skewY": None,
        }
        transform_regex = (
            r"(matrix|scale|translate|rotate|skewX|skewY)\(([0-9e\.\-\s\,]+)\)"
        )

        transform_list.reverse()
        for transform_str in transform_list:
            for t in re.findall(transform_regex, transform_str):
                transform_key = t[0]
                transform_params = [float(p) for p in re.findall(r"[0-9e\-\.]+", t[1])]

                transforms[transform_key](transform_params)

    def save_current_curve(self) -> None:
        self.apply_transforms()
        self.curves.append(copy.deepcopy(self.current_curve))
        self.current_curve = []

    def move(self, p: list) -> None:
        self.current_curve.append(p)

    def move_relative(self, p: list) -> None:
        p = [self.curves[-1][-1][0] + p[0], self.curves[-1][-1][1] + p[1]]
        self.move(p)

    def line(self, p: list) -> None:
        self.current_curve.append(
            [
                self.current_curve[-1][0] + (p[0] - self.current_curve[-1][0]) * 0.33,
                self.current_curve[-1][1] + (p[1] - self.current_curve[-1][1]) * 0.33,
            ]
        )
        self.current_curve.append(
            [
                self.current_curve[-1][0] + (p[0] - self.current_curve[-1][0]) * 0.66,
                self.current_curve[-1][1] + (p[1] - self.current_curve[-1][1]) * 0.66,
            ]
        )
        self.current_curve.append(p)

    def line_relative(self, p: list) -> None:
        p = [self.current_curve[-2] + p[0], self.current_curve[-1] + p[1]]
        self.line(p)

    def horizontal(self, p: list) -> None:
        p = [p[0], self.current_curve[-1][1]]
        self.line(p)

    def horizontal_relative(self, p: list) -> None:
        p = [p[0] + self.current_curve[-1][0]]
        self.horizontal(p)

    def vertical(self, p: list) -> None:
        p = [self.current_curve[-1][0], p[0]]
        self.line(p)

    def vertical_relative(self, p: list) -> None:
        p = [p[0] + self.current_curve[-1][1]]
        self.vertical(p)

    def close_path(self, _: list) -> None:
        if self.current_curve[0] != self.current_curve[-1]:
            self.line(self.current_curve[0])

        self.save_current_curve()

    def cubic_bezier(self, p: list) -> None:
        self.current_curve.append([p[0], p[1]])
        self.current_curve.append([p[2], p[3]])
        self.current_curve.append([p[4], p[5]])

    def cubic_bezier_relative(self, p: list) -> None:
        p = [
            p[0] + self.current_curve[-1][0],
            p[1] + self.current_curve[-1][1],
            p[2] + self.current_curve[-1][0],
            p[3] + self.current_curve[-1][1],
            p[4] + self.current_curve[-1][0],
            p[5] + self.current_curve[-1][1],
        ]
        self.cubic_bezier(p)

    def smooth_cubic_bezier(self, p: list) -> None:
        if self.last_command is not None and self.last_command.lower() in "cs":
            p = [
                self.current_curve[-1][0]
                + (self.current_curve[-1][0] - self.current_curve[-2][0]),
                self.current_curve[-2][1]
                + (self.current_curve[-1][1] - self.current_curve[-2][1]),
                p[1],
                p[2],
                p[3],
                p[4],
            ]
        else:
            p = [
                self.current_curve[-1][0],
                self.current_curve[-1][1],
                p[1],
                p[2],
                p[3],
                p[4],
            ]
        self.cubic_bezier(p)

    def smooth_cubic_bezier_relative(self, p: list) -> None:
        p = [
            p[0] + self.current_curve[-1][0],
            p[1] + self.current_curve[-1][1],
            p[2] + self.current_curve[-1][0],
            p[3] + self.current_curve[-1][1],
        ]
        self.smooth_cubic_bezier(p)

    def quadratic_bezier(self, p: list) -> None:
        p = [p[0], p[1], p[0], p[1], p[2], p[3]]
        self.cubic_bezier(p)

    def quadratic_bezier_relative(self, p: list) -> None:
        p = [
            p[0] + self.current_curve[-1][0],
            p[1] + self.current_curve[-1][1],
            p[2] + self.current_curve[-1][0],
            p[3] + self.current_curve[-1][1],
        ]
        self.quadratic_bezier(p)

    def smooth_quadratic_bezier(self, p: list) -> None:
        if self.last_command is not None and self.last_command.lower() in "qt":
            p = [
                self.current_curve[-1][0]
                + (self.current_curve[-1][0] - self.current_curve[-2][0]),
                self.current_curve[-2][1]
                + (self.current_curve[-1][1] - self.current_curve[-2][1]),
                self.current_curve[-1][0]
                + (self.current_curve[-1][0] - self.current_curve[-2][0]),
                self.current_curve[-2][1]
                + (self.current_curve[-1][1] - self.current_curve[-2][1]),
                p[1],
                p[2],
            ]
        else:
            p = [
                self.current_curve[-1][0],
                self.current_curve[-1][1],
                self.current_curve[-1][0],
                self.current_curve[-1][1],
                p[1],
                p[2],
            ]
        self.cubic_bezier(p)

    def smooth_quadratic_bezier_relative(self, p: list) -> None:
        p = [
            p[0] + self.current_curve[-1][0],
            p[1] + self.current_curve[-1][1],
        ]
        self.smooth_quadratic_bezier(p)

    def convert_path(self, d: str, transform: str | None) -> None:

        commands = {
            "M": self.move,
            "m": self.move_relative,
            "L": self.line,
            "l": self.line_relative,
            "H": self.horizontal,
            "h": self.horizontal_relative,
            "V": self.vertical,
            "v": self.vertical_relative,
            "Z": self.close_path,
            "z": self.close_path,
            "C": self.cubic_bezier,
            "c": self.cubic_bezier_relative,
            "Q": self.quadratic_bezier,
            "q": self.quadratic_bezier_relative,
            "S": self.smooth_cubic_bezier,
            "s": self.smooth_cubic_bezier_relative,
            "T": self.smooth_quadratic_bezier,
            "t": self.smooth_quadratic_bezier_relative,
            # "A": pass,
            # "a": pass,
        }
        command_regex = r"[MmLlHhVvZzCcQqSsTtAa][0-9e\.\,\-\s]*"

        for command_str in re.findall(command_regex, d):
            params = [float(p) for p in re.findall(r"[0-9e\-\.]+", command_str[1:])]
            command = command_str[0]

            if command not in commands.keys():
                raise Exception(f"Unknown command {command}")

            commands[command](params)
            self.last_command = command

        if len(self.current_curve) > 0:
            # Curve not closed
            self.save_current_curve()

    def convert_rect(self, x: float, y: float, w: float, h: float) -> None:
        self.move((x, y))
        self.line((x + w, y))
        self.line((x + w, y + h))
        self.line((x, y + h))
        self.close_path((x, y))

    def convert_circle(self, cx: float, cy: float, r: float) -> None:
        c = 4 / 3 * (math.sqrt(2) - 1)
        self.move((cx + r, cy))
        self.cubic_bezier((cx + r, cy - r * c, cx + r * c, cy - r, cx, cy - r))
        self.cubic_bezier((cx - r * c, cy - r, cx - r, cy - r * c, cx - r, cy))
        self.cubic_bezier((cx - r, cy + r * c, cx - r * c, cy + r, cx, cy + r))
        self.cubic_bezier((cx + r * c, cy + r, cx + r, cy + r * c, cx + r, cy))
        self.close_path([])

    def convert_defs(self, element: ElementTree.Element) -> None:
        if len(element) == 0:
            print(f"Empty def element found: {element}")
            return

        raise Exception("Non-empty def element not handled")

    def interpolate(self) -> None:
        for i, curve in enumerate(self.curves):
            if (len(curve) - 1) % 3 != 0:
                raise Exception("Curve points are not proveided successively!")

            new_curve = []
            for j in range(0, len(curve) - 1, 3):
                p0, p1, p2, p3 = curve[j : j + 4]  # start control1 control2 end

                # Make interpolation pounts dependent on total euclidean distance from p0 > p1 > p2 > p3
                d = max(
                    3,
                    min(
                        100,
                        round(
                            math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
                            + math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                            + math.sqrt((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2)
                        )
                        // 5,
                    ),
                )

                new_curve.append(p0)
                for t in range(d):
                    t = t / d
                    p01 = [
                        (p1[0] - p0[0]) * t + p0[0],
                        (p1[1] - p0[1]) * t + p0[1],
                    ]
                    p12 = [
                        (p2[0] - p1[0]) * t + p1[0],
                        (p2[1] - p1[1]) * t + p1[1],
                    ]
                    p23 = [
                        (p3[0] - p2[0]) * t + p2[0],
                        (p3[1] - p2[1]) * t + p2[1],
                    ]

                    p0112 = [
                        (p12[0] - p01[0]) * t + p01[0],
                        (p12[1] - p01[1]) * t + p01[1],
                    ]
                    p1223 = [
                        (p23[0] - p12[0]) * t + p12[0],
                        (p23[1] - p12[1]) * t + p12[1],
                    ]
                    p01121223 = [
                        (p1223[0] - p0112[0]) * t + p0112[0],
                        (p1223[1] - p0112[1]) * t + p0112[1],
                    ]
                    new_curve.append(p01121223)
            new_curve.append(p3)
            self.curves[i] = new_curve

    def get_min_max(self) -> tuple[float]:
        all_x = [v[0] for curve in self.curves for v in curve]
        all_y = [v[1] for curve in self.curves for v in curve]

        return min(all_x), min(all_y), max(all_x), max(all_y)

    def offset_by(self, x_offset: float, y_offset: float) -> None:
        for i, curve in enumerate(self.curves):
            for j, (x, y) in enumerate(curve):
                self.curves[i][j] = [x + x_offset, y + y_offset]

    def scale_by(self, x_scale: float, y_scale: float) -> None:
        for i, curve in enumerate(self.curves):
            for j, (x, y) in enumerate(curve):
                self.curves[i][j] = [x * x_scale, y * y_scale]

    def scale_to_fit(
        self,
        scale_format: str | tuple[float],
        in_portrait: bool = True,
        use_viewbox: bool = True,
    ) -> None:
        formats = {
            "a0": (841, 1189),
            "a1": (594, 841),
            "a2": (420, 594),
            "a3": (297, 420),
            "a4": (210, 297),
            "a5": (148, 210),
            "a6": (105, 148),
            "a7": (74, 105),
            "a8": (52, 74),
            "a9": (37, 52),
            "a10": (26, 37),
        }

        if isinstance(scale_format, str):
            scale_format = scale_format.lower()

            if scale_format not in formats.keys():
                raise Exception(f"Unknown scale format: {scale_format}")

            if in_portrait:
                w_target, h_target = formats[scale_format]
            else:
                h_target, w_target = formats[scale_format]
        else:
            w_target, h_target = scale_format

        aspect_target = w_target / h_target

        if use_viewbox:
            min_x, min_y, max_x, max_y = self.viewbox
        else:
            min_x, min_y, max_x, max_y = self.get_min_max()
            # Make sure content is at position (0, 0)
            self.offset_by(-min_x, -min_y)

        w_source = max_x - min_x
        h_source = max_y - min_y

        aspect_src = w_source / h_source
        if aspect_src < aspect_target:
            # Source taller than target
            scale = h_target / h_source
        else:
            # Source wider than target
            scale = w_target / w_source
        self.scale_by(scale, scale)

    def optimize_curves(self) -> None:
        optimized_curves = [self.curves.pop(0)]

        while len(self.curves) > 0:

            last_end = optimized_curves[-1][-1]

            # Find closest start/end curve

            nn_dist: int | None = None
            nn_idx: int | None = None
            use_start: bool | None = None

            for i, curve in enumerate(self.curves):
                curve_start = curve[0]
                curve_end = curve[-1]

                dist_to_start = math.sqrt(
                    (curve_start[0] - last_end[0]) ** 2
                    + (curve_start[1] - last_end[1]) ** 2
                )
                if nn_dist is None or nn_dist > dist_to_start:
                    nn_dist = dist_to_start
                    nn_idx = i
                    use_start = True

                dist_to_end = math.sqrt(
                    (curve_end[0] - last_end[0]) ** 2
                    + (curve_end[1] - last_end[1]) ** 2
                )
                if nn_dist is None or nn_dist > dist_to_end:
                    nn_dist = dist_to_end
                    nn_idx = i
                    use_start = False

            # Append curve based on closest start/end and reverse if necessary

            curve = self.curves.pop(nn_idx)
            if not use_start:
                curve.reverse()

            if nn_dist > 0:
                # Append new curve if not continuing at same location
                optimized_curves.append(curve)
            else:
                # Combine curves if continuing at same location
                optimized_curves[-1] += curve[1:]

        self.curves = optimized_curves
