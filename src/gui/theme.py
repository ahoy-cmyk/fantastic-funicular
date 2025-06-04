"""Enhanced theme configuration for Neuromancer."""

from kivy.metrics import dp


class NeuromancerTheme:
    """Professional theme configuration for the application."""

    # Color palette
    PRIMARY_BLUE = (0.2, 0.6, 1.0, 1.0)
    ACCENT_CYAN = (0.0, 0.8, 1.0, 1.0)
    BACKGROUND_DARK = (0.08, 0.08, 0.08, 1.0)
    SURFACE_DARK = (0.12, 0.12, 0.12, 1.0)
    CARD_DARK = (0.15, 0.15, 0.15, 1.0)

    # Text colors
    TEXT_PRIMARY = (1.0, 1.0, 1.0, 1.0)
    TEXT_SECONDARY = (0.8, 0.8, 0.8, 1.0)
    TEXT_HINT = (0.6, 0.6, 0.6, 1.0)

    # Status colors
    SUCCESS_GREEN = (0.2, 0.8, 0.4, 1.0)
    WARNING_ORANGE = (1.0, 0.6, 0.2, 1.0)
    ERROR_RED = (0.9, 0.3, 0.3, 1.0)

    @staticmethod
    def apply_theme(app):
        """Apply the professional theme to the app."""
        theme = app.theme_cls

        # Base theme
        theme.theme_style = "Dark"
        theme.material_style = "M3"

        # Primary colors (use predefined palettes)
        theme.primary_palette = "Blue"
        theme.accent_palette = "Cyan"

        # Set primary hue for better color control
        theme.primary_hue = "700"
        theme.accent_hue = "400"


class UIConstants:
    """UI constants for consistent spacing and sizing."""

    # Spacing
    SPACING_TINY = dp(4)
    SPACING_SMALL = dp(8)
    SPACING_MEDIUM = dp(16)
    SPACING_LARGE = dp(24)
    SPACING_HUGE = dp(32)

    # Padding
    PADDING_SMALL = dp(8)
    PADDING_MEDIUM = dp(16)
    PADDING_LARGE = dp(24)

    # Component sizes
    BUTTON_HEIGHT = dp(40)
    INPUT_HEIGHT = dp(48)
    TOOLBAR_HEIGHT = dp(56)
    FAB_SIZE = dp(56)

    # Elevation
    ELEVATION_CARD = 2
    ELEVATION_DIALOG = 8
    ELEVATION_FAB = 6

    # Border radius
    RADIUS_SMALL = dp(8)
    RADIUS_MEDIUM = dp(12)
    RADIUS_LARGE = dp(16)

    # Animation durations
    ANIMATION_FAST = 0.2
    ANIMATION_MEDIUM = 0.3
    ANIMATION_SLOW = 0.5


class MessageTheme:
    """Theme constants for message cards."""

    # User message colors
    USER_BG = (0.2, 0.6, 1.0, 0.15)
    USER_BORDER = (0.2, 0.6, 1.0, 0.3)

    # Assistant message colors
    ASSISTANT_BG = (0.15, 0.15, 0.15, 1.0)
    ASSISTANT_BORDER = (0.3, 0.3, 0.3, 0.5)

    # System message colors
    SYSTEM_BG = (0.6, 0.6, 0.6, 0.1)
    SYSTEM_BORDER = (0.6, 0.6, 0.6, 0.3)

    # Sizes
    AVATAR_SIZE = dp(32)
    MIN_HEIGHT = dp(60)
    MAX_WIDTH_RATIO = 0.8


class IconTheme:
    """Icon constants for consistent iconography."""

    # Navigation icons
    MENU = "menu"
    BACK = "arrow-left"
    CLOSE = "close"

    # Action icons
    SEND = "send"
    ATTACH = "paperclip"
    VOICE = "microphone"
    COPY = "content-copy"
    EXPORT = "download"
    DELETE = "delete"
    EDIT = "pencil"

    # Status icons
    SUCCESS = "check-circle"
    WARNING = "alert-circle"
    ERROR = "close-circle"
    INFO = "information"

    # Provider icons
    OLLAMA = "robot"
    OPENAI = "brain"
    LMSTUDIO = "desktop-classic"

    # Feature icons
    MEMORY = "brain"
    ANALYTICS = "chart-line"
    SETTINGS = "cog"
    SEARCH = "magnify"
    FILTER = "filter"
