import serial
from tqdm import tqdm


class Plotter:
    def __init__(
        self,
        port: str,
        home: bool = False,
        feed_speed: float = 5000,
        down_dist: float = 5,
    ) -> None:
        """Initializes a Plotter instance to communicate over serial

        Args:
            port (str): The name of the serial port to which the plotter is connected.
            home (bool, optional): Homes the plotter automatically upon connection. Defaults to False.
            feed_speed (float, optional): Movement speed when plotting. Defaults to 5000.
            down_dist (float, optional): _description_. Defaults to 5.

        Raises:
            Exception: _description_
        """
        self.ser = serial.Serial(port, 115200, timeout=0.5)

        self.feed_speed = feed_speed
        self.down_dist = down_dist
        self.gcode_list = []

        if not home:
            return

        if not self.home():
            raise Exception("Coulld not home the printer!")

    @property
    def gcode(self) -> str:
        return "\n".join(self.gcode_list)

    def save_gcode(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            f.write(self.gcode)

    def convert_curves(self, curves: list, home: bool = False) -> None:
        self.gcode_list = []

        if home:
            self.gcode_list.append("$H; Homes plotter")

        self.gcode_list.append("G92 X0 Y0; Set zero position")
        self.gcode_list.append("G21; Set mm units")
        self.gcode_list.append(
            "G90; Absolute positioning - G91 would be relative positioning"
        )
        self.gcode_list.append("$1=255; Keep servo motors on")

        for curve in curves:
            for i, (x, y) in enumerate(curve):

                if i == 0:
                    self.gcode_list.append(
                        f"G0 X{round(x, 2)} Y{round(-y, 2)}; Move to start of path"
                    )
                    self.gcode_list.append("G0 Z5; Move pen down")

                else:
                    self.gcode_list.append(
                        f"G1 F4000 X{round(x, 2)} Y{round(-y, 2)}; Draw"
                    )

            self.gcode_list.append("G0 Z0; Move pen up")

        self.gcode_list.append("$1=0; Turn servo motors off")
        self.gcode_list.append("G0 X0 Y0; Go back home")

    def exec_command(self, cmd: str, tries: int = 10) -> bool:
        """Executes a single C-Code command

        Args:
            cmd (str): C-Code command
            tries (int, optional): Numer of retries before giving up. Defaults to 10.

        Returns:
            bool: True if success
        """
        if cmd[0] == "$":  # Wait for idle when changing GRBL settings
            in_idle = False
            while True:
                self.ser.write(b"?\n")  # Query for status
                r = ""
                while len(r) == 0:
                    r = self.ser.readline()

                # Once idle is detected ...
                if not in_idle and r[1:5] != b"Idle":
                    continue

                # ... confirm idle and ...
                in_idle = True

                # ... wait for command acknowledgement
                if r != b"ok\r\n":
                    continue

                break

        self.ser.write(f"{cmd}\n".encode("utf-8"))
        while True:
            if tries > 0:
                r = ""
                while len(r) == 0:
                    r = self.ser.readline()
                if r == b"ok\r\n":
                    return True
                tries -= 1
            else:
                print(r)
                return False

    def exec_commands(self, cmds: str | None = None) -> bool:
        """Executes a passed string of G-Code commands and uses the `self.gcode` if `None` is passed.

        Args:
            cmds (str | None, optional): G-Code commands. Defaults to None.

        Returns:
            bool: True if success
        """
        if cmds is None:
            cmds = self.gcode

        for cmd in tqdm(cmds.split("\n")):

            # Skip empty lines
            if len(cmd) == 0:
                continue

            # Extract comment
            cmt = ""
            if ";" in cmd:
                cmd, cmt = cmd.split(";")
                cmt = cmt.strip()

            # Format command
            cmd = cmd.strip()
            if not self.exec_command(cmd):
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
            G0 Z{self.down_dist}; Move pen down
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
            f"G1 F{self.feed_speed} X{round(x, 2)} Y{round(-y, 2)}"
        )
