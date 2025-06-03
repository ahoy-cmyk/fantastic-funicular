"""Compatibility layer for KivyMD 2.0."""

from kivymd.uix.button import MDButton, MDButtonText


class MDRaisedButton(MDButton):
    """Compatibility wrapper for MDRaisedButton."""

    def __init__(self, text="", **kwargs):
        # Extract button-specific kwargs
        on_release = kwargs.pop("on_release", None)
        on_press = kwargs.pop("on_press", None)
        pos_hint = kwargs.pop("pos_hint", None)

        super().__init__(**kwargs)

        # Set button style
        self.style = "elevated"

        # Add text
        if text:
            self.add_widget(MDButtonText(text=text))

        # Set callbacks
        if on_release:
            self.bind(on_release=on_release)
        if on_press:
            self.bind(on_press=on_press)

        # Set position hint
        if pos_hint:
            self.pos_hint = pos_hint


class MDFlatButton(MDButton):
    """Compatibility wrapper for MDFlatButton."""

    def __init__(self, text="", **kwargs):
        # Extract button-specific kwargs
        on_release = kwargs.pop("on_release", None)
        on_press = kwargs.pop("on_press", None)

        super().__init__(**kwargs)

        # Set button style
        self.style = "text"

        # Add text
        if text:
            self.add_widget(MDButtonText(text=text))

        # Set callbacks
        if on_release:
            self.bind(on_release=on_release)
        if on_press:
            self.bind(on_press=on_press)
