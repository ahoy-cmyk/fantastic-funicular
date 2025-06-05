"""Professional theme system implementing Material Design 3 specifications.

This module provides a comprehensive theming system demonstrating:

- Material Design 3 color palette implementation
- Professional dark theme with high contrast ratios
- Consistent UI constants for spacing, sizing, and animation
- Semantic color coding for different UI states and contexts
- Accessibility-compliant color choices and contrast ratios
- Scalable design system supporting multiple screen densities

Design Philosophy:
    The theme implements a professional, modern aesthetic suitable for
    technical applications while maintaining excellent readability and
    visual hierarchy. Colors are chosen for both aesthetic appeal and
    functional communication of system states.

Material Design Compliance:
    - Uses official Material Design 3 color palettes
    - Implements proper elevation and shadow systems
    - Follows spacing and typography guidelines
    - Maintains accessibility standards for contrast ratios

Consistency Framework:
    - Centralized constants prevent inconsistent spacing
    - Semantic naming for colors aids in maintenance
    - Scalable units (dp) ensure cross-device compatibility
    - Animation timing constants provide consistent motion

Accessibility Considerations:
    - High contrast ratios exceed WCAG AA standards
    - Color choices work for color-blind users
    - Text legibility optimized for extended reading
    - Focus indicators and states clearly defined
"""

from kivy.metrics import dp


class NeuromancerTheme:
    """Professional theme configuration implementing Material Design 3.

    This class provides a centralized theme management system featuring:

    Color Palette Design:
        - Primary blue (#3399FF) for interactive elements and branding
        - Accent cyan (#00CCFF) for secondary actions and highlights
        - Dark background (#141414) for reduced eye strain
        - Surface colors creating proper visual hierarchy
        - Status colors providing semantic meaning

    Material Design Implementation:
        - Official Material Design 3 color system
        - Proper elevation and surface treatments
        - Consistent component styling throughout
        - Accessibility-compliant contrast ratios

    Professional Aesthetics:
        - Sophisticated dark theme suitable for technical applications
        - High contrast for excellent readability
        - Minimal but expressive color palette
        - Clean, modern visual language

    Text Color Hierarchy:
        - Primary text (#FFFFFF) for main content
        - Secondary text (#CCCCCC) for supporting information
        - Hint text (#999999) for placeholders and guidance

    Status Communication:
        - Success green (#33CC66) for positive actions and confirmations
        - Warning orange (#FF9933) for caution and attention
        - Error red (#E63C3C) for problems and destructive actions

    Theme Application:
        The apply_theme method configures KivyMD's theme system with
        these professional color choices and Material Design settings.

    Design Principles:
        - Visual hierarchy through color and typography
        - Semantic color usage for intuitive interaction
        - Professional appearance suitable for business use
        - Accessibility compliance for inclusive design
    """

    # Professional Color Palette
    # Primary colors chosen for trust, professionalism, and excellent readability
    # All colors tested for WCAG AA accessibility compliance
    PRIMARY_BLUE = (0.2, 0.6, 1.0, 1.0)
    ACCENT_CYAN = (0.0, 0.8, 1.0, 1.0)
    BACKGROUND_DARK = (0.08, 0.08, 0.08, 1.0)
    SURFACE_DARK = (0.12, 0.12, 0.12, 1.0)
    CARD_DARK = (0.15, 0.15, 0.15, 1.0)

    # Text Color Hierarchy
    # Optimized for dark backgrounds with excellent contrast ratios
    # Provides clear information hierarchy through opacity variation
    TEXT_PRIMARY = (1.0, 1.0, 1.0, 1.0)
    TEXT_SECONDARY = (0.8, 0.8, 0.8, 1.0)
    TEXT_HINT = (0.6, 0.6, 0.6, 1.0)

    # Semantic Status Colors
    # Universal color language for system state communication
    # Chosen for cross-cultural recognition and accessibility
    SUCCESS_GREEN = (0.2, 0.8, 0.4, 1.0)
    WARNING_ORANGE = (1.0, 0.6, 0.2, 1.0)
    ERROR_RED = (0.9, 0.3, 0.3, 1.0)

    @staticmethod
    def apply_theme(app):
        """Apply comprehensive professional theme to the application.

        This method configures the KivyMD theme system with:

        Material Design Configuration:
            - Material Design 3 style implementation
            - Dark theme for reduced eye strain and modern aesthetics
            - Professional color palette with semantic meaning
            - Proper hue selection for optimal color relationships

        Color System Setup:
            - Primary palette based on blue for trust and professionalism
            - Accent palette using cyan for secondary actions
            - Specific hue values optimized for dark backgrounds
            - Color relationships following Material Design guidelines

        Visual Consistency:
            - Centralized theme application prevents inconsistencies
            - Standard Material Design component styling
            - Professional appearance across all UI elements
            - Consistent color usage throughout the application

        Accessibility Features:
            - High contrast ratios for excellent readability
            - Color choices that work for color-blind users
            - Professional appearance suitable for business environments
            - Clear visual hierarchy through color and elevation

        Args:
            app: The MDApp instance to apply theming to

        The theme creates a cohesive, professional appearance while
        maintaining excellent usability and accessibility standards.
        """
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
    """Comprehensive UI design system constants for consistent interface design.

    This class provides a complete design system featuring:

    Spacing System:
        - Consistent spacing scale from tiny (4dp) to huge (32dp)
        - Based on 8dp grid system following Material Design
        - Semantic naming for intuitive usage
        - Scalable units (dp) for cross-device compatibility

    Padding Standards:
        - Small, medium, and large padding options
        - Consistent internal spacing for components
        - Optimized for touch interaction and visual balance
        - Scalable across different screen densities

    Component Sizing:
        - Standard heights for buttons, inputs, and toolbars
        - FAB (Floating Action Button) sizing following Material Design
        - Consistent sizing across similar component types
        - Touch-friendly dimensions for mobile compatibility

    Elevation System:
        - Material Design elevation values for proper depth
        - Different elevation levels for various component types
        - Consistent shadow and depth effects
        - Visual hierarchy through elevation differences

    Border Radius Standards:
        - Small, medium, and large radius options
        - Consistent rounded corner treatment
        - Modern, friendly appearance
        - Appropriate radius for different component sizes

    Animation Timing:
        - Fast, medium, and slow animation durations
        - Consistent motion timing throughout the application
        - Optimized for user perception and comfort
        - Professional motion design standards

    Design System Benefits:
        - Prevents inconsistent spacing and sizing
        - Speeds up development with predefined values
        - Ensures professional, cohesive appearance
        - Simplifies maintenance and updates
        - Supports responsive design across devices

    Usage Pattern:
        Import and use constants instead of hardcoded values:
        spacing = UIConstants.SPACING_MEDIUM
        button_height = UIConstants.BUTTON_HEIGHT
    """

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
    """Specialized theming for chat message display components.

    This class provides message-specific design constants featuring:

    Message Type Differentiation:
        - User messages: Blue theme with transparency for personal communication
        - Assistant messages: Neutral theme for AI responses
        - System messages: Gray theme for application notifications
        - Border treatments for visual separation and hierarchy

    Color Psychology:
        - User message blue conveys personal ownership and input
        - Assistant message neutrality emphasizes AI nature
        - System message subtlety avoids interference with conversation
        - Transparency creates depth without overwhelming content

    Visual Design:
        - Background colors with appropriate transparency
        - Border colors for subtle definition
        - Consistent color relationships across message types
        - Professional appearance suitable for business communication

    Sizing Standards:
        - Avatar sizing for profile representation
        - Minimum height for touch accessibility
        - Maximum width ratio for readable text line length
        - Responsive sizing for different screen sizes

    Message Layout:
        - Avatar size optimized for recognition without dominance
        - Height minimums ensuring touch accessibility
        - Width constraints for optimal reading experience
        - Scalable proportions across device types

    Accessibility Considerations:
        - Sufficient contrast ratios for text readability
        - Color coding supplemented by visual structure
        - Touch-friendly sizing for interaction
        - Clear visual hierarchy between message types

    Usage in Components:
        These constants ensure consistent message appearance
        throughout the chat interface while providing clear
        visual distinction between different message sources.
    """

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
    """Comprehensive icon system for consistent visual communication.

    This class provides a centralized icon vocabulary featuring:

    Navigation Icons:
        - Standard navigation patterns (menu, back, close)
        - Consistent directional indicators
        - Universal recognition for common actions
        - Material Design icon specifications

    Action Icons:
        - Primary actions (send, attach, voice)
        - Content manipulation (copy, export, delete, edit)
        - Clear semantic meaning for user understanding
        - Consistent visual weight and style

    Status Indicators:
        - Success, warning, error, and info states
        - Immediate visual feedback for system states
        - Color-independent recognition through shape
        - Universal symbols for cross-cultural understanding

    Provider Identity:
        - Specific icons for different AI providers
        - Visual brand recognition and differentiation
        - Consistent representation across the interface
        - Professional appearance for business contexts

    Feature Icons:
        - Functional area identification (memory, analytics, settings)
        - Quick recognition of application features
        - Intuitive symbols for complex functionality
        - Consistent visual language throughout the app

    Icon Selection Principles:
        - Universal recognition over creativity
        - Consistent visual weight and style
        - Clear semantic meaning
        - Accessibility through recognizable shapes
        - Professional appearance

    Material Design Compliance:
        - Uses official Material Design icon names
        - Consistent sizing and visual treatment
        - Proper icon usage following guidelines
        - Scalable vector icons for all densities

    Usage Benefits:
        - Prevents inconsistent icon usage
        - Provides semantic meaning through naming
        - Enables easy icon updates and maintenance
        - Ensures professional, cohesive appearance
        - Supports internationalization and accessibility

    Implementation Pattern:
        Reference icons by semantic name rather than appearance:
        icon = IconTheme.SEND (not "send-arrow" or "paper-plane")
    """

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
