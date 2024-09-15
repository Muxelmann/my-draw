# My-Draw

With this project I intend to control my Pen Plotter (i.e. Bachin T-A4).

It appears to respond to [GRBL](https://github.com/grbl/grbl) commands, but I did not find an adequate SVG to G-Code converter. So I wrote my own.

## Project Structure

The root directory containing the code is `src`. The `my_draw` module inside `src` provides a `Parser` class and a `Plotter` class.

### tl;dr

The following code is a basic use case:

```python
from my_draw import Parser, Plotter

# Load image
parser = Parser.from_file("image.svg")
# Interpolate (optional)

parser.interpolate()
# Optimize curves (optional)
parser.optimize_curves()
# Scale to fit paper (optional)
parser.scale_to_fit("a5")

# Connect to plotter
plotter = Plotter("/dev/cu.usbserial")
# Convert curves to G-Code
plotter.convert_curves(parser.curves)
# Execute converted G-Code
plotter.exec_commands()
```

### Parser

This class allows you to extract all geometries from an SVG file and convert them into a list of "*curves*" that are only a series of points the pen should follow to draw the geometries. It allows interpolation and optimization for better accuracy and improved drawing speed. More details below.

#### SVG Format

The SVG file content is well documented by the *mdn web docs* [^1] comprises several elements. Here only the following elements are taken into account:

[^1]: [Developer Mozilla: SVG File](https://developer.mozilla.org/en-US/docs/Web/SVG)

- `g`: a grouping of several elements - often a transformation or style is shared among the grouped elements.
- `rect`: a definition of a rectangle element.
- `circle`: a definition of a circle element.
- `path`: a definition of a path element.

**TODO:** properly implement `defs` and `use` elements although this appears not yet necessary for SVGs exported from Affinity and p5js.

The above elements are converted into a representation based on a series of Bézier curves. E.g., the four sides of a `rect` are converted into four subsequent Bézier curves, a `circle` is approximated by 4 Bézier curves each forming an arc using a standard approximation[^2] where $ c = \frac{4}{3} \cdot (\sqrt{2} - 1) $, and a `path` is converted into multiple Bézier curves based on its `d` attribute[^3] as defined.

[^2]: [Approximate a circle with cubic Bézier curves](https://web.archive.org/web/20240415180204/https://spencermortensen.com/articles/bezier-circle/)
[^3]: [Developer Mozilla: `d` attribute](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/d)

As a result, each curve in the list of curves is essentially structured like so:

```python
(x0, y0), # Start point of 1st curve
(x1, y1), # 1st control point of 1st curve
(x2, y2), # 2nd control point of 1st curve
(x3, y3), # End point of 1st curve / Start point of 2nd curve
(x4, y4), # 1st control point of 2nd curve
...
(xN, yN)  # End point of Nth curve
```

Where, the number of points `N` is:

$$ 3 \times \text{No. of Bézier curves} + 1$$

These curves can be drawn by passing the curves to the `Plotter` class as explained below. For improved plotting quality and speed, the interpolation and optimization methods are provided by the `Parser` class.

##### Interpolation

Because the Bézier points of each curve (especially the control points) rarely lie on the actual path of the Bézier curve, it may be necessary to interpolate them to better approximate the path to be plotted. To this extend, the `Parser` class interpolates into something between 3 and 100 points depending on how long the Bézier curve is estimated to be. Because each Bézier curve is defined by exactly four points, it will be drawn as a cubic Bézier curve (animation from Wikipedia[^4]).

[^4]: [Wikipedia: Bézier curve](https://en.wikipedia.org/wiki/Bézier_curve)

![](https://upload.wikimedia.org/wikipedia/commons/d/db/Bézier_3_big.gif)

To estimate the point at each time $t$, the points $P_0$ to $P_3$ are treated as vectors $\vec{p}_0$ to $\vec{p}_3$. Here, $\vec{p}_i = (p_{i,x}, p_{i,y})$ where $i \in [0, ..., 3]$:

$$
\begin{align}
\vec{p}_{01} &= \vec{p}_{0} + (\vec{p}_{1} - \vec{p}_{0}) \cdot t\\
\vec{p}_{12} &= \vec{p}_{1} + (\vec{p}_{2} - \vec{p}_{1}) \cdot t\\
\vec{p}_{23} &= \vec{p}_{2} + (\vec{p}_{3} - \vec{p}_{2}) \cdot t\\
\vec{p}_{0112} &= \vec{p}_{01} + (\vec{p}_{12} - \vec{p}_{01}) \cdot t\\
\vec{p}_{1223} &= \vec{p}_{12} + (\vec{p}_{23} - \vec{p}_{12}) \cdot t\\
\vec{p}_{01121223} &= \vec{p}_{0112} + (\vec{p}_{1223} - \vec{p}_{0112}) \cdot t
\end{align}
$$

Here, the point represented by $\vec{p}_{01121223}$ is a point lying on the Bézier curve. For long Bézier curves, the number of interpolated points is high (i.e. 100 at most) and for short Bézier curves the number of interpolated points is low (i.e. 3 at least).

##### Optimization

Some programs (e.g. p5js) export a complex shape as an SVG containing a plurality of simple paths, each being a single straight line. Because these lines are short and the number of them is large, the plotted shape comes out fine. However, this way of plotting would result in the plotter raising and lowering the pen often at the same point between plotting each simple line. To avoid this problem, the `Parser` can optimize the curves by joining:

1. reordering them to make sure the pen needs not travel far between the end of one curve and the beginning of the next curve (even reversing the direction of curves if necessary), and
2. joining curves when the end point of one curve is the same as the start point of the next curve.

Although the current reordering algorithm uses a basic nearest neighbor (NN) search, it results in a significantly improved plotting speed.

##### Scaling

For plotting each pixel dimension represents a millimeter dimension. That means, a line of `1 px` is drawn as a line of `1 mm` by the plotter. Therefore, if the provided SVG is too big for the plotter, it needs to be scaled down. If the paper format matches DIN, it can be scaled to fit using the `Parser`'s scaling function e.g., to an `"a5"` format. Alternatively, width and height can be provided if a non-standard format is used.

### Plotter

The `Plotter` class connects to the plotter using a serial interface and sends GRBL commands to the plotter for controlling it.

Plotting speed (a.k.a. `feed speed`) and pen travel (i.e. how far to move the pen along the Z-axis to start and end plotting) can be customized.

The `Plotter` class has a method that converts the curves provided by the `Parser` class for plotting. GRBL v1.1 commands[^5] [^6] are quite extensive, but I found at least the following commands to be useful for plotting:

[^5]: [Github: Grbl v1.1 Configuration](https://github.com/gnea/grbl/wiki/Grbl-v1.1-Configuration)
[^6]: [Sainsmart: Grbl V1.1 Quick Reference](https://web.archive.org/web/20240415202628/https://www.sainsmart.com/blogs/news/grbl-v1-1-quick-reference)

- `$H`: Run Homing Cycle
- `?`: Status report query
- `$1=0`: Turn servo motors off i.e. de-energized
- `$1=255`: Keep servo motors on i.e. energized
- `G92 X0 Y0`: Set zero position
- `G21`: Set units to mm
- `G90`: All distances and positions are absolute values
- `G91`: All distances and positions are relative values
- `G0 X<val> Y<val>`: Rapid position change to X and Y coordinate
- `G1 F<val> X<val> Y<val>`: Straight line move at given speed (feed rate `F`) to X and Y
- `G0 Z<val>`: Raises and lowers pen e.g. by setting 0 or 5 mm, respectively

It is noteworthy, that the plotter has an inverted Y-axis. So either invert the direction[^7] using:

[^7]: [Direction port invert, mask](https://github.com/gnea/grbl/wiki/Grbl-v1.1-Configuration#3--direction-port-invert-mask)

```
$3=2; Invert Y only - mask is b'010' for XYZ"
```

Or make sure the Y values are inverted when sending them to the plotter.

##### Troubleshooting

###### Error 8

Sometimes when changing a setting, an error is returned by the plotter instead of `b"ok\r\n"`. Usually this is error code 8, indicating an error because a `$` command is only valid when the plotter is in `Idle` state. To this end, an state checking loop precedes command sending stage, only for `$` commands.

###### ~~Lost command~~

Also, sometimes the plotter does not receive or acknowledge a command. To address this issue, a command is repeatedly sent a certain number of tries with a delay between tries.

**Update**: This appears no longer necessary in view of fix for double `ok\r\n` below.

###### Double OK

When a command is acknowledged by GRBL, a `b"ok\r\n"` is returned. However, it turns out, GRBL interprets a command followed by both carriage return `\r` and new line `\n` as two commands and hence returns two OKs i.e., `b"ok\r\nok\r\n"`[^8]. By only terminating commands only with `\n`, this issue is resolved and reattempting command sending is no longer necessary.

[^8]: [Github](https://github.com/grbl/grbl/issues/1024)
