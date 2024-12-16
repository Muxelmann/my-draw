import serial
from tqdm import tqdm
import math


class Plotter:
    def __init__(
        self,
        port: str,
        home: bool = False,
        feed_speed: float = 5000,
        down_dist: float = 5,
        x_angle_error: float = 0,
        keep_on: bool = True,
    ) -> None:
        """Initializes a Plotter instance to communicate over serial

        Args:
            port (str): The name of the serial port to which the plotter is connected.
            home (bool, optional): Homes the plotter automatically upon connection. Defaults to False.
            feed_speed (float, optional): Movement speed when plotting. Defaults to 5000.
            down_dist (float, optional): Distance the pen moves up and down. Defaults to 5.
            x_angle_error (float, optional): Correct for skew because x/y angle is not 90n deg. Defaults to 0.
            keep_on (bool, optional): Keep the motors on during plotting. Defaults to True
        Raises:
            Exception: _description_
        """
        self.ser = serial.Serial(port, 115200, timeout=0.5)

        self._feed_speed = feed_speed
        self._down_dist = down_dist
        self._gcode_list = []
        self.x_angle_error = x_angle_error
        self.keep_on = keep_on

        if not home:
            return

        if not self.home():
            raise Exception("Coulld not home the printer!")

    @property
    def gcode(self) -> str:
        return "\n".join(self._gcode_list)

    def save_gcode(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            f.write(self.gcode)

    def init_gcode(self, home: bool = False) -> None:
        self._gcode_list = []
        if home:
            self._gcode_list.append("$H; Homes plotter")

        self._gcode_list.append("G92 X0 Y0; Set zero position")
        self._gcode_list.append("G21; Set mm units")
        self._gcode_list.append(
            "G90; Absolute positioning - G91 would be relative positioning"
        )
        self._gcode_list.append("G0 Z0; Move pen up")
        if self.keep_on:
            self._gcode_list.append("$1=255; Keep servo motors on")

    def finish_gcode(self) -> None:
        if self.keep_on:
            self._gcode_list.append("$1=0; Turn servo motors off")
        self._gcode_list.append("G0 X0 Y0; Go back home")

    def convert_curves(self, curves: list) -> None:

        for curve in curves:
            for i, (x, y) in enumerate(curve):

                if self.x_angle_error != 0:
                    angle = self.x_angle_error * math.pi / 180
                    y += x * math.sin(angle)
                    x += x * (1 - math.cos(angle))

                if i == 0:
                    self._gcode_list.append(
                        f"G1 F{self._feed_speed} X{round(x, 2)} Y{round(-y, 2)}; Move to start of path"
                    )
                    self._gcode_list.append("G0 Z5; Move pen down")

                else:
                    self._gcode_list.append(
                        f"G1 F{self._feed_speed} X{round(x, 2)} Y{round(-y, 2)}; Draw"
                    )

            self._gcode_list.append("G0 Z0; Move pen up")

    def exec_command(self, cmd: str) -> bool:
        """Executes a single C-Code command
        and makes sure that the printer is idle before executing a `$`-command
        for changing GRBL settings.

        Args:
            cmd (str): C-Code command

        Returns:
            bool: True if success
        """
        if cmd[0] == "$":  # Wait for idle when changing GRBL settings

            # Check if idle
            in_idle = False
            # Check if queried for status
            queried = False

            while True:
                # Query for status if query not acknowledged
                if not queried and not in_idle:
                    self.ser.write(b"?\n")
                    queried = True

                # If queried and idle is not yet confirmed
                elif queried and not in_idle:

                    # Read / Wait for response and ...
                    r = ""
                    while len(r) == 0:
                        r = self.ser.readline()

                    # ... check if idle
                    if r.startswith(b"<Idle"):
                        # ... confirm idle
                        in_idle = True
                        continue

                    # ... check if running
                    elif r.startswith(b"<Run"):
                        # ... wait for idle again
                        continue

                    # ... check if query is concluded
                    elif r == b"ok\r\n":
                        queried = False
                        continue

                    # ... unknown response
                    else:
                        print(f"Unexpected response: {r}")
                        return False

                elif queried and in_idle:
                    r = ""
                    while len(r) == 0:
                        r = self.ser.readline()

                    # ... check if idle
                    if r == b"ok\r\n":
                        queried = False
                        continue

                # elif not queried and in_idle: # Equivalent to `else:`
                else:
                    # If idle has been confirmed leave while loop
                    break

        self.ser.write(f"{cmd}\n".encode("utf-8"))  # Send command

        # Obtain ok response after command execution
        r = ""
        while len(r) == 0:
            r = self.ser.readline()
        if r == b"ok\r\n":
            return True

        return False

    def exec_commands(self, commands: str | None = None) -> bool:
        """Executes a passed string of G-Code commands and uses the `self.gcode` if `None` is passed.

        Args:
            commands (str | None, optional): G-Code commands. Defaults to None.

        Returns:
            bool: True if success
        """
        if commands is None:
            commands = self.gcode

        for command in tqdm(commands.split("\n")):
            command = command.strip()

            # Skip empty lines
            if len(command) == 0:
                continue

            # Extract comment
            comment = ""
            if ";" in command:
                command, comment = command.split(";")
                comment = comment.strip()

            # Format command
            command = command.strip()
            if not self.exec_command(command):
                return False
        return True

    def home(self) -> bool:
        """Homes the plotter

        It also sets the zero positions for X and Y, sets Millimeter as units and absolute positioning

        Returns:
            bool: True if success
        """
        if self.exec_command("$H"):
            return self.exec_commands(
                """
                G92 X0 Y0; Set zero position
                G21; Set mm units
                G90; Absolute positioning - G91 would be relative positioning
                """
            )
        else:
            print("Error homing")
            return False

    def pen_down(self) -> bool:
        """Moves the pen down

        Returns:
            bool: True if success
        """
        return self.exec_commands(
            f"""
            $1=255; Keep stepper motors on ("M84 S0" for Marlin)
            G0 Z{self._down_dist}; Move pen down
            """
        )

    def pen_up(self) -> bool:
        """Moves the pen down

        Returns:
            bool: True if success
        """
        return self.exec_commands(
            f"""
            G0 Z1; Move pen down
            $1=0; Keep stepper motors off ("M84 S0" for Marlin)
            """
        )

    def move(self, x: float, y: float) -> bool:
        return self.exec_commands(f"G0 X{round(x, 2)} Y{round(-y, 2)}")

    def draw(self, x: float, y: float) -> bool:
        return self.exec_command(
            f"G1 F{self._feed_speed} X{round(x, 2)} Y{round(-y, 2)}"
        )
