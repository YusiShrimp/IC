import tkinter as tk
import numpy as np


class DataConverterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bitwise Data Converter")
        self.geometry("640x720")

        self._data = np.uint16(42)

        self.columnconfigure(0, minsize=400, weight=1)

        self.control_frame = tk.Frame(self, relief=tk.RAISED, borderwidth=1, width=750)
        self.control_frame.columnconfigure(0, minsize=300, weight=1)
        self.control_frame.columnconfigure(1, minsize=300, weight=1)

        self.representations = {}
        self.variables = {}
        self.converters = {}
        self.deconverters = {}

        self.ivcmd = (self.register(self.on_invalid),)

        self._add_representation("bin", "binary (variable length)", lambda: str(bin(self._data)[2:]),
                                 lambda x: np.uint16(int(x, 2)))
        self._add_representation("bin_fw", "binary (fixed width)", lambda: str(np.binary_repr(self._data, 16)),
                                 lambda x: np.uint16(int(x, 2)))
        self._add_representation("oct", "octal", lambda: str(oct(self._data)[2:]),
                                 lambda x: np.int16(int(x, 8)))
        self._add_representation("dec", "decimal", lambda: str(np.int32(self._data)),
                                 lambda x: np.uint16(int(x)))
        self._add_representation("hex", "hexadecimal", lambda: str(hex(self._data)[2:]),
                                 lambda x: np.uint16(int(x, 16)))
        self._add_representation("real", "floating point (16bit)", lambda: str(self._data.view(np.float16)),
                                 lambda x: np.uint16(int.from_bytes(np.float16(x).tobytes(), "little")))
        self._add_representation("ascii", "ascii", lambda: "".join([chr(b) for b in self._bdata]),
                                 lambda x: np.frombuffer(bytes(x, 'ascii'), dtype=np.uint8).view(np.uint16))

        self.control_frame.pack(pady=10)

    @property
    def _bdata(self):
        """Return the byte representation of the data."""
        return [int(b) for b in self._data.tobytes()]

    def _add_representation(self, short_name, name, get_value_func, set_value_func):
        """Add a new representation to the control frame."""
        self.variables[short_name] = tk.StringVar(value=get_value_func())
        self.converters[short_name] = get_value_func
        self.deconverters[short_name] = set_value_func

        entry = tk.Entry(
            self.control_frame,
            name=short_name,
            width=20,
            justify="right",
            textvariable=self.variables[short_name],
            validate="key",
            validatecommand=(self.register(self.validate_data), "%W", "%P"),
        )
        label = tk.Label(self.control_frame, text=name.capitalize())
        label.grid(column=0, row=len(self.representations), padx=5, pady=5)
        entry.grid(column=1, row=len(self.representations), padx=5, pady=5)
        self.representations[short_name] = entry

    def _update_representations(self, name):

        for rname, var in self.variables.items():
            if rname == name:
                continue
            try:
                var.set(self.converters[rname]())
            except Exception as e:
                print(f"Error updating representation {rname}: {e}")

    def validate_data(self, name, value):
        if len(value) > 16:
            return False
        widget = self.nametowidget(name)
        if value == "":
            value = "0"  # Allow empty input to be treated as zero
        name = name.split(".")[-1]
        if value == self.converters[name]():
            return True
        print(self._data, name, value, self.deconverters)
        # Get the last part of the name to match the representation
        if name in self.deconverters:
            try:
                self._data = self.deconverters[name](value)
                self._update_representations(name)
                widget.config(bg="white")
                return True
            except ValueError as e:
                print(f"Error converting {value} for {name}: {e}")
                return False
        return True

    def on_invalid(self, *args):

        """Handle invalid input by turning the entry red."""

        widget = self.nametowidget(args[0])
        widget.config(bg="red")
        return False


if __name__ == "__main__":
    app = DataConverterApp()
    app.mainloop()
