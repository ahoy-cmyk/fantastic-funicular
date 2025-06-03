"""Comprehensive theming and aesthetic fixes for the enhanced chat interface."""

from kivy.metrics import dp
from kivymd.uix.card import MDCard

# Define consistent color scheme
THEME_COLORS = {
    # Primary colors
    "background": (0.05, 0.05, 0.1, 1),  # Dark blue-black background
    "surface": (0.08, 0.08, 0.12, 1),  # Slightly lighter surface
    "primary": (0.2, 0.5, 0.8, 1),  # Blue primary color
    "secondary": (0.8, 0.2, 0.5, 1),  # Pink secondary color
    # Message bubbles
    "user_bubble": (0.15, 0.1, 0.2, 1),  # Purple-ish user messages
    "assistant_bubble": (0.1, 0.1, 0.15, 1),  # Blue-ish assistant messages
    "system_bubble": (0.12, 0.12, 0.12, 1),  # Gray system messages
    # Special elements
    "memory_bubble": (0.9, 0.4, 0.7, 0.9),  # Bright pink for memory context
    "drawer_bg": (0.05, 0.05, 0.1, 0.95),  # Drawer background
    "toolbar_bg": (0.08, 0.08, 0.13, 1),  # Toolbar background
    # Text colors
    "text_primary": (1, 1, 1, 1),  # White primary text
    "text_secondary": (0.7, 0.7, 0.7, 1),  # Gray secondary text
    "text_hint": (0.5, 0.5, 0.5, 1),  # Darker hint text
}

# Define spacing and sizing constants
SPACING = {
    "small": dp(5),
    "medium": dp(10),
    "large": dp(20),
    "message_padding": dp(15),
    "drawer_width": dp(300),
    "toolbar_height": dp(56),
    "input_height": dp(60),
    "memory_bubble_height": dp(120),
}

# Z-index elevation values
ELEVATION = {
    "card": 1,
    "toolbar": 2,
    "memory_bubble": 4,
    "drawer": 16,
    "dialog": 24,
}


def apply_chat_theme_fixes(enhanced_chat_screen):
    """Apply comprehensive theming fixes to the enhanced chat screen."""

    # 1. Fix navigation drawer colors and positioning
    if hasattr(enhanced_chat_screen, "nav_drawer"):
        drawer = enhanced_chat_screen.nav_drawer
        drawer.md_bg_color = THEME_COLORS["drawer_bg"]
        drawer.scrim_color = (0, 0, 0, 0.7)  # Darker scrim
        drawer.elevation = ELEVATION["drawer"]
        drawer.radius = (0, 16, 16, 0)
        # Ensure drawer width is consistent
        if hasattr(drawer, "width"):
            drawer.width = SPACING["drawer_width"]

    # 2. Fix toolbar styling
    if hasattr(enhanced_chat_screen, "toolbar"):
        toolbar = enhanced_chat_screen.toolbar
        toolbar.md_bg_color = THEME_COLORS["toolbar_bg"]
        toolbar.elevation = ELEVATION["toolbar"]
        toolbar.specific_text_color = THEME_COLORS["text_primary"]

    # 3. Fix memory context bubble
    if hasattr(enhanced_chat_screen, "memory_context"):
        memory = enhanced_chat_screen.memory_context
        memory.md_bg_color = THEME_COLORS["memory_bubble"]
        memory.elevation = ELEVATION["memory_bubble"]
        memory.padding = SPACING["medium"]

    # 4. Fix message input area
    if hasattr(enhanced_chat_screen, "message_input"):
        input_field = enhanced_chat_screen.message_input
        input_field.hint_text_color_normal = THEME_COLORS["text_hint"]
        input_field.text_color_normal = THEME_COLORS["text_primary"]
        input_field.line_color_normal = THEME_COLORS["primary"]
        input_field.line_color_focus = THEME_COLORS["secondary"]

    # 5. Fix chat messages area background
    if hasattr(enhanced_chat_screen, "messages_scroll"):
        scroll = enhanced_chat_screen.messages_scroll
        scroll.bar_color = THEME_COLORS["primary"]
        scroll.bar_inactive_color = THEME_COLORS["surface"]
        scroll.bar_width = dp(4)

    # 6. Apply consistent spacing
    if hasattr(enhanced_chat_screen, "messages_layout"):
        layout = enhanced_chat_screen.messages_layout
        layout.spacing = SPACING["medium"]
        layout.padding = SPACING["medium"]


def create_styled_message_card(content: str, role: str, timestamp: str = None) -> MDCard:
    """Create a properly styled message card with consistent theming."""

    # Choose bubble color based on role
    bubble_colors = {
        "user": THEME_COLORS["user_bubble"],
        "assistant": THEME_COLORS["assistant_bubble"],
        "system": THEME_COLORS["system_bubble"],
    }

    card = MDCard(
        orientation="vertical",
        size_hint_y=None,
        adaptive_height=True,
        padding=SPACING["message_padding"],
        spacing=SPACING["small"],
        elevation=ELEVATION["card"],
        md_bg_color=bubble_colors.get(role, THEME_COLORS["surface"]),
        radius=[dp(16), dp(16), dp(16), dp(16)],  # Rounded corners
    )

    # Add shadow for depth
    card.shadow_offset = (0, 2)
    card.shadow_color = (0, 0, 0, 0.3)
    card.shadow_softness = 4

    return card


def fix_overlapping_elements(screen):
    """Fix any overlapping UI elements."""

    # Ensure proper widget hierarchy
    widgets = screen.children[:]

    # Sort widgets by their intended z-order
    z_order = {
        "MDBoxLayout": 0,  # Base content
        "MDScrollView": 0,  # Scrollable areas
        "MDTopAppBar": 1,  # Toolbar
        "MemoryContextCard": 2,  # Memory bubble
        "MDNavigationDrawer": 3,  # Drawer on top
    }

    # Re-add widgets in correct order
    screen.clear_widgets()

    sorted_widgets = sorted(widgets, key=lambda w: z_order.get(type(w).__name__, 0))
    for widget in sorted_widgets:
        screen.add_widget(widget)


def apply_responsive_sizing(screen):
    """Apply responsive sizing to prevent overlaps on different screen sizes."""
    from kivy.core.window import Window

    # Adjust message bubble max width based on screen size
    max_bubble_width = min(Window.width * 0.7, dp(500))

    # Adjust drawer width for smaller screens
    if Window.width < dp(600):
        drawer_width = Window.width * 0.8
    else:
        drawer_width = SPACING["drawer_width"]

    if hasattr(screen, "nav_drawer"):
        screen.nav_drawer.width = drawer_width

    return max_bubble_width
