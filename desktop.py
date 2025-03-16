#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenManus Desktop Application
A Windows desktop application for the OpenManus agent using GTK.
"""

# Platform detection
import os
import sys
import logging
import threading
import tomli
import tomli_w  # For writing TOML files
import asyncio
import ctypes
from typing import Optional, Dict, Any, List

# GUI toolkit imports
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

# Environment setup
from dotenv import load_dotenv
load_dotenv(override=True)  # Force reload environment variables from .env file

# OpenManus imports
from app.agent.manus import Manus
from app.logger import logger

# Windows-specific configuration
if os.name == 'nt':
    # Force basic theme and backend
    os.environ["GTK_THEME"] = "win32"
    os.environ["GDK_BACKEND"] = "win32"

    # High DPI fix
    ctypes.windll.shcore.SetProcessDpiAwareness(1)


class GtkLogHandler(logging.Handler):
    """Custom log handler that redirects log messages to a GTK TextView"""

    def __init__(self, text_view):
        super().__init__()
        self.text_view = text_view
        self.buffer = text_view.get_buffer()
        self.formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')

    def emit(self, record):
        log_entry = self.formatter.format(record) + '\n'

        # Use GLib.idle_add to safely update the UI from any thread
        GLib.idle_add(self.update_text_view, log_entry)

    def update_text_view(self, text):
        end_iter = self.buffer.get_end_iter()
        self.buffer.insert(end_iter, text)
        # Auto-scroll to the end
        self.text_view.scroll_to_iter(self.buffer.get_end_iter(), 0.0, False, 0.0, 0.0)
        return False  # Required for GLib.idle_add


class ManusApp(Gtk.Application):
    """
    Main application class for OpenManus desktop application.
    Manages application-level functionality, configuration, and client initialization.
    """
    def __init__(self):
        super().__init__(application_id='com.openmanus.desktop')

        # Initialize configuration parameters
        self.config_params = {}

        # Initialize agent
        self.agent = None

        # Initialize background tasks
        self.running_tasks = []

    def do_activate(self):
        """Create main window when application is activated"""
        self.main_window = MainWindow(self)
        self.main_window.show_all()
        self.main_window.present()  # Force focus

    def initialize_agent(self):
        """Initialize the AI agent with current settings"""
        if not hasattr(self, 'main_window') or self.main_window is None:
            logger.error("Main window not initialized")
            return

        provider = self.main_window.provider_combo.get_active_text().lower()
        model = self.main_window.model_combo.get_active_text()

        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.toml')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file missing: {config_path}")

            with open(config_path, 'rb') as f:
                config = tomli.load(f)

            # Config structure validation
            if 'llm' not in config:
                config['llm'] = {}
            if 'default' not in config['llm']:
                config['llm']['default'] = {
                    'model': 'gpt-3.5-turbo',
                    'temperature': 0.7,
                    'max_tokens': 2048
                }

            if 'llm' not in config:
                raise ValueError(f"No configuration found for llm")

            llm_config = config['llm']

            self.agent = Manus(
                provider=provider,
                api_key=llm_config.get('default', {}).get('api_key', ''),
                model=model,
                temperature=self.main_window.temp_scale.get_value(),
                max_tokens=int(self.main_window.tokens_scale.get_value()),
                base_url=llm_config.get('default', {}).get('base_url'),
                api_version=llm_config.get('default', {}).get('api_version')
            )

            # Vision configuration
            if config.get('vision', {}).get('enabled', False):
                self.agent.enable_vision(
                    model=config['vision'].get('model', 'gpt-4-vision-preview'),
                    max_tokens=config['vision'].get('max_tokens', 2048)
                )
                self.main_window.vision_switch.set_active(True)

            # Initialize tool lists
            available_tools = []
            enabled_tools = []
            
            if hasattr(self.agent, 'enable_tools'):
                available_tools = self.agent.get_available_tools()
                enabled_tools = [
                    t.lower().replace(' ', '_') 
                    for t in config.get('tools', {}).get('enabled', [])
                    if t.lower().replace(' ', '_') in available_tools
                ]
                self.agent.enable_tools(enabled_tools)

                # Update UI checkboxes with display names
                for display_name, checkbox in self.main_window.tool_switches:
                    tool_id = display_name.lower().replace(' ', '_')
                    checkbox.set_active(tool_id in enabled_tools)
                    checkbox.set_visible(tool_id in available_tools)

            # Initialize UI state
            self.main_window.model_combo.set_active_id(config['llm']['default']['model'])
            self.main_window.temp_scale.set_value(config['llm']['default']['temperature'])
            self.main_window.tokens_scale.set_value(config['llm']['default']['max_tokens'])

            for tool, checkbox in self.main_window.tool_switches:
                checkbox.set_active(tool in enabled_tools)
                checkbox.set_visible(tool in available_tools)

        except KeyError as e:
            logging.error(f"Missing required config key: {str(e)}")
            self.main_window.show_error_dialog(f"Invalid configuration: {str(e)}")
        except Exception as e:
            logging.error(f"Configuration error: {str(e)}")
            self.main_window.show_error_dialog(f"Failed to initialize agent: {str(e)}")
        return self.agent

    def run_background_task(self, task_function, callback, *args, **kwargs):
        """Run a task in the background to keep UI responsive"""
        def run_async_task():
            # Create a new event loop for the thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Run the coroutine in the new event loop
                result = loop.run_until_complete(task_function(*args, **kwargs))
                # Schedule the callback to run in the main thread
                GLib.idle_add(lambda: callback(result))
            except Exception as e:
                # Schedule error handling in the main thread
                GLib.idle_add(lambda: callback(None, error=str(e)))
            finally:
                loop.close()

        # Start the thread
        thread = threading.Thread(target=run_async_task)
        thread.daemon = True
        thread.start()
        self.running_tasks.append(thread)
        return thread


class MainWindow(Gtk.ApplicationWindow):
    """
    Main window class for OpenManus desktop application.
    Handles the UI components, user interactions, and display logic.
    """
    def __init__(self, app):
        super().__init__(title="OpenManus Desktop", application=app)
        self.app = app
        self.set_default_size(900, 700)
        self.set_position(Gtk.WindowPosition.CENTER)

        # Lifecycle logging
        self.connect('map', lambda _: print("Window mapped"))
        self.connect('unmap', lambda _: print("Window unmapped"))
        self.connect('destroy', lambda _: print("Window destroyed"))

        # Set application icon
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.jpg")
            if os.path.exists(icon_path):
                self.set_icon_from_file(icon_path)
        except Exception as e:
            logger.warning(f"Could not load application icon: {e}")

        # Initialize all UI components first
        self.build_main_window()
        self.build_settings_dialog()

        # Migrate legacy config if needed
        self.migrate_legacy_config()

        # Load config into UI after initialization
        GLib.idle_add(self.load_config_into_ui)

        # Connect signals
        self.connect("delete-event", self.on_close)

    def migrate_legacy_config(self):
        """Ensure config has required fields"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.toml')
            if not os.path.exists(config_path):
                # Create basic config file with correct structure
                basic_config = {
                    'llm': {
                        'default': {
                            'model': 'gpt-4o',
                            'base_url': 'https://api.openai.com/v1',
                            'api_key': '',
                            'max_tokens': 4096,
                            'temperature': 1.0,
                            'api_type': 'Openai',
                            'api_version': ''
                        }
                    }
                }

                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'wb') as f:
                    tomli_w.dump(basic_config, f)
                return

            # Read existing config
            with open(config_path, 'rb') as f:
                config = tomli.load(f)

            modified = False

            # Ensure llm section exists
            if 'llm' not in config:
                config['llm'] = {}
                modified = True

            # Ensure llm.default section exists
            if 'default' not in config['llm']:
                # If there are direct settings in llm, move them to default
                default_settings = {}
                direct_keys = ['model', 'base_url', 'api_key', 'max_tokens', 'temperature', 'api_type', 'api_version']

                for key in direct_keys:
                    if key in config['llm']:
                        default_settings[key] = config['llm'][key]
                        del config['llm'][key]

                # Add missing required fields
                required_fields = {
                    'model': 'gpt-4o',
                    'base_url': 'https://api.openai.com/v1',
                    'api_key': '',
                    'max_tokens': 4096,
                    'temperature': 1.0,
                    'api_type': 'Openai',
                    'api_version': ''
                }

                for field, default_value in required_fields.items():
                    if field not in default_settings or default_settings[field] is None:
                        default_settings[field] = default_value

                config['llm']['default'] = default_settings
                modified = True
            else:
                # Ensure required fields exist in llm.default
                required_fields = {
                    'model': 'gpt-4o',
                    'base_url': 'https://api.openai.com/v1',
                    'api_key': '',
                    'max_tokens': 4096,
                    'temperature': 1.0,
                    'api_type': 'Openai',
                    'api_version': ''
                }

                for field, default_value in required_fields.items():
                    if field not in config['llm']['default'] or config['llm']['default'][field] is None:
                        config['llm']['default'][field] = default_value
                        modified = True

            if modified:
                with open(config_path, 'wb') as f:
                    tomli_w.dump(config, f)
                logging.info("Updated config structure")

        except Exception as e:
            logging.error(f"Error ensuring config fields: {str(e)}")

    def build_main_window(self):
        """Build the main UI components"""
        # Main container
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(main_box)

        # Add header bar with title and controls
        header = self.build_header()
        main_box.pack_start(header, False, False, 0)

        # Main paned container for sidebar and content
        self.main_paned = Gtk.Paned.new(Gtk.Orientation.HORIZONTAL)
        main_box.pack_start(self.main_paned, True, True, 0)

        # Build sidebar
        self.sidebar = self.build_sidebar()
        self.main_paned.pack1(self.sidebar, False, False)

        # Build main content area
        self.content_area = self.build_content_area()
        self.main_paned.pack2(self.content_area, True, True)

        # Set initial pane position (30% for sidebar)
        self.main_paned.set_position(270)

        # Add status bar
        self.status_bar = Gtk.Statusbar()
        self.status_context = self.status_bar.get_context_id("main")
        main_box.pack_end(self.status_bar, False, False, 0)
        self.status_bar.push(self.status_context, "Ready")

    def build_header(self):
        """Build the header bar with controls"""
        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header_box.set_margin_top(6)
        header_box.set_margin_bottom(6)
        header_box.set_margin_start(10)
        header_box.set_margin_end(10)

        # Logo and title
        try:
            logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.jpg")
            if os.path.exists(logo_path):
                pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(logo_path, 32, 32, True)
                logo = Gtk.Image.new_from_pixbuf(pixbuf)
                header_box.pack_start(logo, False, False, 0)
        except Exception as e:
            logger.warning(f"Could not load logo: {e}")

        # Create empty labels first, then set markup
        title = Gtk.Label()
        title.set_markup("<b>OpenManus</b>")
        title.set_margin_start(10)
        header_box.pack_start(title, False, False, 0)

        # Spacer
        spacer = Gtk.Label()
        header_box.pack_start(spacer, True, True, 0)

        # Settings button
        settings_button = Gtk.Button.new_from_icon_name("preferences-system", Gtk.IconSize.BUTTON)
        settings_button.set_tooltip_text("Settings")
        settings_button.connect("clicked", self.on_settings_clicked)
        header_box.pack_end(settings_button, False, False, 0)

        return header_box

    def build_sidebar(self):
        """Build the sidebar with configuration options"""
        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        sidebar_box.set_margin_top(10)
        sidebar_box.set_margin_bottom(10)
        sidebar_box.set_margin_start(10)
        sidebar_box.set_margin_end(10)

        # Sidebar header
        # Create empty labels first, then set markup
        sidebar_header = Gtk.Label()
        sidebar_header.set_markup("<b>Configuration</b>")
        sidebar_header.set_halign(Gtk.Align.START)
        sidebar_box.pack_start(sidebar_header, False, False, 0)

        # Separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        sidebar_box.pack_start(separator, False, False, 10)

        # Provider selection
        provider_frame = Gtk.Frame(label="AI Provider")
        provider_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        provider_frame.add(provider_box)

        self.provider_combo = Gtk.ComboBoxText()
        providers = ["openai", "anthropic", "mistral", "groq", "ollama"]
        for provider in providers:
            self.provider_combo.append_text(provider.capitalize())
        self.provider_combo.set_active(0)
        self.provider_combo.connect("changed", self.on_provider_changed)
        provider_box.pack_start(self.provider_combo, False, False, 5)

        sidebar_box.pack_start(provider_frame, False, False, 10)

        # Model selection now depends on provider
        self.model_combo = Gtk.ComboBoxText()
        sidebar_box.pack_start(self.model_combo, False, False, 5)

        # Temperature setting
        temp_label = Gtk.Label(label="Temperature:")
        temp_label.set_halign(Gtk.Align.START)
        sidebar_box.pack_start(temp_label, False, False, 10)

        temp_adj = Gtk.Adjustment(
            value=0.0,
            lower=0.0,
            upper=2.0,
            step_increment=0.1,
            page_increment=0.5,
            page_size=0
        )
        self.temp_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=temp_adj)
        self.temp_scale.set_digits(1)
        self.temp_scale.set_value_pos(Gtk.PositionType.RIGHT)
        self.temp_scale.connect("value-changed", self.on_temperature_changed)
        sidebar_box.pack_start(self.temp_scale, False, False, 0)

        # Max tokens setting
        tokens_label = Gtk.Label(label="Max Tokens:")
        tokens_label.set_halign(Gtk.Align.START)
        sidebar_box.pack_start(tokens_label, False, False, 10)

        tokens_adj = Gtk.Adjustment(
            value=4096,
            lower=256,
            upper=16384,
            step_increment=256,
            page_increment=1024,
            page_size=0
        )
        self.tokens_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=tokens_adj)
        self.tokens_scale.set_digits(0)
        self.tokens_scale.set_value_pos(Gtk.PositionType.RIGHT)
        self.tokens_scale.connect("value-changed", self.on_tokens_changed)
        sidebar_box.pack_start(self.tokens_scale, False, False, 0)

        # Vision settings
        vision_frame = Gtk.Frame(label="Vision Settings")
        vision_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        vision_frame.add(vision_box)

        self.vision_switch = Gtk.Switch()
        self.vision_switch.set_active(False)
        vision_box.pack_start(Gtk.Label(label="Enable Vision"), False, False, 0)
        vision_box.pack_start(self.vision_switch, False, False, 0)

        sidebar_box.pack_start(vision_frame, False, False, 10)

        # Browser configuration
        browser_frame = Gtk.Frame(label="Browser Settings")
        browser_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        browser_frame.add(browser_box)

        self.headless_switch = Gtk.Switch()
        browser_box.pack_start(self._create_labeled_control("Headless mode:", self.headless_switch), False, False, 5)

        self.chrome_path_entry = Gtk.Entry()
        browser_box.pack_start(self._create_labeled_control("Chrome path:", self.chrome_path_entry), False, False, 5)

        self.wss_entry = Gtk.Entry()
        browser_box.pack_start(self._create_labeled_control("WSS URL:", self.wss_entry), False, False, 5)

        self.cdp_entry = Gtk.Entry()
        browser_box.pack_start(self._create_labeled_control("CDP URL:", self.cdp_entry), False, False, 5)

        sidebar_box.pack_start(browser_frame, False, False, 10)

        # Tool selection
        tools_label = Gtk.Label(label="Available Tools:")
        tools_label.set_halign(Gtk.Align.START)
        sidebar_box.pack_start(tools_label, False, False, 10)

        # Tool checkboxes
        tools_frame = Gtk.Frame()
        tools_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        tools_frame.add(tools_box)

        tool_names = ["Python Execute", "Web Search", "Browser Use", "File Saver"]
        self.tool_switches = []
        for tool in tool_names:
            checkbox = Gtk.CheckButton.new_with_label(tool)
            checkbox.set_active(True)
            checkbox.connect("toggled", self.on_tool_toggled, tool)
            tools_box.pack_start(checkbox, False, False, 0)
            self.tool_switches.append((tool, checkbox))

        sidebar_box.pack_start(tools_frame, False, False, 0)

        # Spacer
        spacer = Gtk.Label()
        sidebar_box.pack_start(spacer, True, True, 0)

        # Run button
        run_button = Gtk.Button.new_with_label("Run Agent")
        run_button.connect("clicked", self.on_run_clicked)
        sidebar_box.pack_end(run_button, False, False, 0)

        return sidebar_box

    def _create_labeled_control(self, label_text, control):
        """Create a box with a label and control widget"""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        label = Gtk.Label(label=label_text)
        label.set_halign(Gtk.Align.START)
        label.set_size_request(120, -1)  # Fixed width for alignment

        box.pack_start(label, False, False, 0)
        box.pack_start(control, True, True, 0)

        return box

    def build_content_area(self):
        """Build the main content area with prompt input and response display"""
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        content_box.set_margin_top(10)
        content_box.set_margin_bottom(10)
        content_box.set_margin_start(10)
        content_box.set_margin_end(10)

        # Prompt input area
        prompt_label = Gtk.Label(label="Enter your prompt:")
        prompt_label.set_halign(Gtk.Align.START)
        content_box.pack_start(prompt_label, False, False, 0)

        prompt_scroll = Gtk.ScrolledWindow()
        prompt_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        prompt_scroll.set_min_content_height(100)

        self.prompt_buffer = Gtk.TextBuffer()
        self.prompt_text = Gtk.TextView.new_with_buffer(self.prompt_buffer)
        self.prompt_text.set_wrap_mode(Gtk.WrapMode.WORD)
        prompt_scroll.add(self.prompt_text)

        content_box.pack_start(prompt_scroll, False, False, 0)

        # Submit button
        self.submit_button = Gtk.Button.new_with_label("Submit")
        self.submit_button.connect("clicked", self.on_submit_clicked)
        content_box.pack_start(self.submit_button, False, False, 0)

        # Response display area
        response_label = Gtk.Label(label="Response:")
        response_label.set_halign(Gtk.Align.START)
        content_box.pack_start(response_label, False, False, 10)

        # Create a scrolled window for the response text
        response_scroll = Gtk.ScrolledWindow()
        response_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        response_scroll.set_min_content_height(200)

        # Create a text view for the response
        self.response_text = Gtk.TextView()
        self.response_text.set_editable(False)
        self.response_text.set_wrap_mode(Gtk.WrapMode.WORD)
        response_scroll.add(self.response_text)
        content_box.pack_start(response_scroll, True, True, 0)

        # Set up custom log handler to capture output
        self.log_handler = GtkLogHandler(self.response_text)
        self.log_handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)

        # Progress indicator
        self.progress_bar = Gtk.ProgressBar()
        self.progress_bar.set_pulse_step(0.1)
        content_box.pack_start(self.progress_bar, False, False, 0)

        return content_box

    def build_settings_dialog(self):
        """Create the settings dialog"""
        self.settings_dialog = Gtk.Dialog(title="Settings", parent=self, flags=0)
        self.settings_dialog.set_default_size(400, 300)

        content_area = self.settings_dialog.get_content_area()
        content_area.set_spacing(10)
        content_area.set_margin_start(10)
        content_area.set_margin_end(10)
        content_area.set_margin_top(10)
        content_area.set_margin_bottom(10)

        # Provider-specific settings container
        self.provider_settings_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        content_area.add(self.provider_settings_container)

        # API Key
        self.api_key_label = Gtk.Label(label="API Key:")
        self.api_key_label.set_halign(Gtk.Align.START)
        self.api_key_entry = Gtk.Entry()
        self.api_key_entry.set_visibility(False)  # Password-style entry
        self.provider_settings_container.add(self.api_key_label)
        self.provider_settings_container.add(self.api_key_entry)

        # Base URL
        self.base_url_label = Gtk.Label(label="API Base URL:")
        self.base_url_label.set_halign(Gtk.Align.START)
        self.base_url_entry = Gtk.Entry()
        self.provider_settings_container.add(self.base_url_label)
        self.provider_settings_container.add(self.base_url_entry)

        # Headless mode
        self.headless_switch = Gtk.Switch()
        self.provider_settings_container.add(self._create_labeled_control("Headless mode:", self.headless_switch))

        # Chrome path
        self.chrome_path_entry = Gtk.Entry()
        self.provider_settings_container.add(self._create_labeled_control("Chrome path:", self.chrome_path_entry))

        # WSS URL
        self.wss_url_entry = Gtk.Entry()
        self.provider_settings_container.add(self._create_labeled_control("WSS URL:", self.wss_url_entry))

        # CDP URL
        self.cdp_url_entry = Gtk.Entry()
        self.provider_settings_container.add(self._create_labeled_control("CDP URL:", self.cdp_url_entry))

        # Buttons
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        button_box.set_halign(Gtk.Align.END)

        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.connect("clicked", lambda w: self.settings_dialog.hide())

        save_button = Gtk.Button(label="Save")
        save_button.connect("clicked", self.on_save_settings)

        button_box.pack_start(cancel_button, False, False, 0)
        button_box.pack_start(save_button, False, False, 0)

        content_area.add(button_box)
        self.settings_dialog.show_all()
        self.settings_dialog.hide()  # Initially hidden

    def load_config_into_ui(self):
        try:
            with open('config/config.toml', 'rb') as f:
                config = tomli.load(f)

            # Load providers
            providers = ["openai", "anthropic", "mistral", "groq", "ollama"]
            self.provider_combo.handler_block_by_func(self.on_provider_changed)
            self.provider_combo.remove_all()
            for provider in providers:
                self.provider_combo.append_text(provider.capitalize())
            self.provider_combo.set_active(0)
            self.provider_combo.handler_unblock_by_func(self.on_provider_changed)

            # Load browser settings
            browser_config = config.get('browser', {})
            self.headless_switch.set_active(browser_config.get('headless', False))
            self.chrome_path_entry.set_text(browser_config.get('chrome_instance_path', ''))
            self.wss_url_entry.set_text(browser_config.get('wss_url', ''))
            self.cdp_url_entry.set_text(browser_config.get('cdp_url', ''))

        except Exception as e:
            logging.error(f"Config loading error: {str(e)}")

    def save_ui_to_config(self):
        # Load existing config to preserve all settings
        try:
            with open('config/config.toml', 'rb') as f:
                config = tomli.load(f)
        except FileNotFoundError:
            config = {'llm': {}, 'browser': {}}

        # Update current provider
        config['llm']['default']['api_key'] = self.api_key_entry.get_text()
        base_url = self.base_url_entry.get_text()
        if base_url:
            config['llm']['default']['base_url'] = base_url

        # Update browser settings
        config['browser'] = {
            'headless': self.headless_switch.get_active(),
            'chrome_instance_path': self.chrome_path_entry.get_text(),
            'wss_url': self.wss_url_entry.get_text(),
            'cdp_url': self.cdp_url_entry.get_text()
        }

        with open('config/config.toml', 'wb') as f:
            tomli_w.dump(config, f)

    def on_model_changed(self, combo):
        """Handle model selection change"""
        model = combo.get_active_text()
        logger.info(f"Model changed to: {model}")
        # Update config
        if self.app.config_params.get('llm') is None:
            self.app.config_params['llm'] = {}
        self.app.config_params['llm']['default']['model'] = model

    def on_provider_changed(self, combo):
        """Handle provider selection change"""
        active_text = combo.get_active_text()
        if active_text is None:
            # No provider selected, add default provider
            self.provider_combo.handler_block_by_func(self.on_provider_changed)
            self.provider_combo.remove_all()
            self.provider_combo.append_text("OpenAI")
            self.provider_combo.set_active(0)
            self.provider_combo.handler_unblock_by_func(self.on_provider_changed)
            return

        provider = active_text.lower()
        models = self._get_provider_models(provider)
        self.model_combo.handler_block_by_func(self.on_model_changed)
        self.model_combo.remove_all()
        if models:
            for model in models:
                self.model_combo.append_text(model)
            # Set first model as default if available
            self.model_combo.set_active(0)
            logger.info(f"Loaded {len(models)} models for {provider}")
        else:
            logger.warning(f"No models found for {provider}, using defaults")
            default_models = ["gpt-3.5-turbo", "gpt-4"]
            for model in default_models:
                self.model_combo.append_text(model)
            self.model_combo.set_active(0)
        self.model_combo.handler_unblock_by_func(self.on_model_changed)

    def _get_provider_models(self, provider):
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.toml')
            with open(config_path, 'rb') as f:
                config = tomli.load(f)

            # Handle legacy config format
            if 'providers' not in config:
                # Convert legacy format to new providers format
                if provider == 'openai' and 'llm' in config:
                    # Return models from llm section if available
                    if 'model' in config['llm']:
                        return [config['llm']['model']]
                    else:
                        return ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
                else:
                    # Default models for other providers
                    default_models = {
                        'anthropic': ["claude-3-opus", "claude-3-sonnet"],
                        'mistral': ["mistral-large", "mistral-medium"],
                        'groq': ["llama3-70b", "llama3-8b"],
                        'ollama': ["llama3", "mistral"]
                    }
                    return default_models.get(provider, [])

            # New config format
            if provider in config['providers']:
                return config['providers'][provider].get('models', [])
            return []

        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            return []

    def on_temperature_changed(self, scale):
        """Handle temperature setting change"""
        temp = scale.get_value()
        logger.info(f"Temperature changed to: {temp}")
        # Update config
        if self.app.config_params.get('llm') is None:
            self.app.config_params['llm'] = {}
        self.app.config_params['llm']['default']['temperature'] = temp

    def on_tokens_changed(self, scale):
        """Handle max tokens setting change"""
        tokens = int(scale.get_value())
        logger.info(f"Max tokens changed to: {tokens}")
        # Update config
        if self.app.config_params.get('llm') is None:
            self.app.config_params['llm'] = {'default': {}}
        self.app.config_params['llm']['default']['max_tokens'] = tokens

    def on_tool_toggled(self, checkbox, tool_name):
        """Handle tool checkbox toggle"""
        active = checkbox.get_active()
        logger.info(f"Tool '{tool_name}' {'enabled' if active else 'disabled'}")
        # Update config (to be implemented)

    def on_settings_clicked(self, widget):
        """Update settings dialog for current provider and show it"""
        current_provider = self.provider_combo.get_active_text()
        self.update_settings_for_provider(current_provider)
        self.settings_dialog.show_all()

    def update_settings_for_provider(self, provider_name):
        """Update settings dialog fields based on selected provider"""
        # Update labels
        self.api_key_label.set_text(f"{provider_name} API Key:")

        # Load settings
        try:
            with open('config/config.toml', 'rb') as f:
                config = tomli.load(f)

            if 'llm' in config:
                self.api_key_entry.set_text(config['llm'].get('default', {}).get('api_key', ''))
                self.base_url_entry.set_text(config['llm'].get('default', {}).get('base_url', self.get_default_base_url(provider_name.lower())))

            if 'browser' in config:
                if 'headless' in config['browser']:
                    self.headless_switch.set_active(config['browser']['headless'])
                if 'chrome_instance_path' in config['browser']:
                    self.chrome_path_entry.set_text(config['browser']['chrome_instance_path'])
                if 'wss_url' in config['browser']:
                    self.wss_url_entry.set_text(config['browser']['wss_url'])
                if 'cdp_url' in config['browser']:
                    self.cdp_url_entry.set_text(config['browser']['cdp_url'])
        except Exception as e:
            logging.error(f"Error loading settings: {str(e)}")

    def get_default_base_url(self, provider):
        """Return default base URL for known providers"""
        defaults = {
            'openai': 'https://api.openai.com/v1',
            'anthropic': 'https://api.anthropic.com',
            'mistral': 'https://api.mistral.ai/v1',
            'groq': 'https://api.groq.com/v1',
            'ollama': 'http://localhost:11434/api'
        }
        return defaults.get(provider, '')

    def on_save_settings(self, button):
        """Save settings"""
        provider = self.provider_combo.get_active_text().lower()

        try:
            # Load existing config
            with open('config/config.toml', 'rb') as f:
                config = tomli.load(f)

            # Update llm section
            if 'llm' not in config:
                config['llm'] = {}

            config['llm']['default']['api_key'] = self.api_key_entry.get_text()
            base_url = self.base_url_entry.get_text()
            if base_url:
                config['llm']['default']['base_url'] = base_url

            # Add defaults if they don't exist
            if 'model' not in config['llm']['default']:
                config['llm']['default']['model'] = 'gpt-4o'
            if 'max_tokens' not in config['llm']['default']:
                config['llm']['default']['max_tokens'] = 4096
            if 'temperature' not in config['llm']['default']:
                config['llm']['default']['temperature'] = 1.0
            if 'api_type' not in config['llm']['default']:
                config['llm']['default']['api_type'] = 'Openai'
            if 'api_version' not in config['llm']['default']:
                config['llm']['default']['api_version'] = ''

            # Update browser section
            if self.headless_switch.get_active() or self.chrome_path_entry.get_text() or self.wss_url_entry.get_text() or self.cdp_url_entry.get_text():
                if 'browser' not in config:
                    config['browser'] = {}

                config['browser']['headless'] = self.headless_switch.get_active()

                chrome_path = self.chrome_path_entry.get_text()
                if chrome_path:
                    config['browser']['chrome_instance_path'] = chrome_path

                wss_url = self.wss_url_entry.get_text()
                if wss_url:
                    config['browser']['wss_url'] = wss_url

                cdp_url = self.cdp_url_entry.get_text()
                if cdp_url:
                    config['browser']['cdp_url'] = cdp_url

            # Save config
            with open('config/config.toml', 'wb') as f:
                tomli_w.dump(config, f)

            self.settings_dialog.hide()
        except Exception as e:
            logging.error(f"Error saving settings: {str(e)}")
            self.show_error_dialog(f"Failed to save settings: {str(e)}")

    def on_run_clicked(self, button):
        """Handle run button click"""
        # Initialize agent if not already initialized
        if self.app.agent is None:
            self.app.initialize_agent()
            self.status_bar.push(self.status_context, "Agent initialized")

    def on_submit_clicked(self, button):
        # Get config overrides from UI
        vision_enabled = self.vision_switch.get_active()

        # Update agent config before execution
        if hasattr(self.app.agent, 'vision_enabled'):
            self.app.agent.vision_enabled = vision_enabled

        # Get selected tools from UI
        tools = [tool[0] for tool in self.tool_switches if tool[1].get_active()]

        # Get prompt from text view
        prompt = self.prompt_text.get_buffer().get_text(
            self.prompt_text.get_buffer().get_start_iter(),
            self.prompt_text.get_buffer().get_end_iter(),
            True
        )

        if not prompt.strip():
            self.show_error_dialog("Please enter a prompt")
            return

        # Clear response
        self.response_text.get_buffer().set_text("")

        # Log the start of agent execution to the response area
        logging.info("Starting agent execution with prompt: %s", prompt[:50] + "..." if len(prompt) > 50 else prompt)

        # Start progress indicator
        self.start_progress_pulse()

        # Initialize agent if needed
        if not hasattr(self.app, 'agent') or self.app.agent is None:
            self.app.initialize_agent()

        # Get values directly from UI widgets
        model = self.model_combo.get_active_text() or "gpt-3.5-turbo"
        temperature = self.temp_scale.get_value()
        max_tokens = int(self.tokens_scale.get_value())

        # Run agent in background
        self.submit_button.set_sensitive(False)
        self._run_agent_thread(prompt, model, temperature, max_tokens, tools)

    async def _run_agent_async(self, prompt, model, temperature, max_tokens, tools):
        """Run the agent asynchronously and capture its output"""
        try:
            # Configure the agent if attributes exist
            if hasattr(self.app.agent, 'model'):
                self.app.agent.model = model
            if hasattr(self.app.agent, 'temperature'):
                self.app.agent.temperature = temperature
            if hasattr(self.app.agent, 'max_tokens'):
                self.app.agent.max_tokens = max_tokens

            # Log configuration
            logging.info(f"Agent configuration: model={model}, temp={temperature}, max_tokens={max_tokens}")
            logging.info(f"Selected tools: {tools}")

            # Run the agent
            response = await self.app.agent.run(prompt)

            # Log and display the final response
            logging.info(f"Agent response: {response}")

            # Update UI on the main thread
            GLib.idle_add(self.update_response_complete, response)

        except Exception as e:
            error_msg = f"Error running agent: {str(e)}"
            logging.error(error_msg)
            GLib.idle_add(self.update_response_error, error_msg)
        finally:
            # Always re-enable the submit button
            GLib.idle_add(self.submit_button.set_sensitive, True)
            GLib.idle_add(self.stop_progress_pulse)

    def _run_agent_thread(self, prompt, model, temperature, max_tokens, tools):
        """Run the agent in a separate thread"""
        # Create and run the asyncio event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_agent_async(prompt, model, temperature, max_tokens, tools))
        finally:
            loop.close()

    def update_response_complete(self, response):
        """Update the UI with the completed response"""
        # Add a separator line
        buffer = self.response_text.get_buffer()
        end_iter = buffer.get_end_iter()
        buffer.insert(end_iter, "\n\n----- FINAL RESPONSE -----\n\n")
        buffer.insert(end_iter, response)

        # Update status
        self.status_bar.push(self.status_context, "Request completed")
        return False  # Required for GLib.idle_add

    def update_response_error(self, error_msg):
        """Update the UI with an error message"""
        buffer = self.response_text.get_buffer()
        end_iter = buffer.get_end_iter()
        buffer.insert(end_iter, f"\n\nERROR: {error_msg}\n")

        # Update status
        self.status_bar.push(self.status_context, "Error occurred")
        return False  # Required for GLib.idle_add

    def start_progress_pulse(self):
        """Start progress bar pulsing animation"""
        self.progress_pulse_active = True

        def update_pulse():
            if self.progress_pulse_active:
                self.progress_bar.pulse()
                return True
            else:
                self.progress_bar.set_fraction(0.0)
                return False

        self.progress_pulse_id = GLib.timeout_add(100, update_pulse)

    def stop_progress_pulse(self):
        """Stop progress bar pulsing animation"""
        self.progress_pulse_active = False

    def show_error_dialog(self, message):
        """Show an error dialog with the given message"""
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Error"
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

    def on_close(self, widget, event):
        """Handle window close event"""
        # Clean up resources
        return False  # Allow the window to close


class SettingsDialog(Gtk.Dialog):
    """Settings dialog for configuring API keys and other settings"""
    def __init__(self, parent):
        super().__init__(
            title="Settings",
            transient_for=parent,
            flags=Gtk.DialogFlags.MODAL | Gtk.DialogFlags.DESTROY_WITH_PARENT
        )
        self.parent = parent
        self.set_default_size(400, 300)

        # Add buttons
        self.add_button("Cancel", Gtk.ResponseType.CANCEL)
        self.add_button("Save", Gtk.ResponseType.OK)

        # Build UI
        self.build_ui()

    def build_ui(self):
        """Build the settings dialog UI"""
        content_area = self.get_content_area()
        content_area.set_margin_top(10)
        content_area.set_margin_bottom(10)
        content_area.set_margin_start(10)
        content_area.set_margin_end(10)
        content_area.set_spacing(10)

        # API Key settings
        api_label = Gtk.Label(label="OpenAI API Key:")
        api_label.set_halign(Gtk.Align.START)
        content_area.pack_start(api_label, False, False, 0)

        self.api_entry = Gtk.Entry()
        self.api_entry.set_visibility(False)  # Hide API key
        self.api_entry.set_placeholder_text("Enter your OpenAI API key")
        content_area.pack_start(self.api_entry, False, False, 0)

        # Base URL settings
        url_label = Gtk.Label(label="API Base URL:")
        url_label.set_halign(Gtk.Align.START)
        content_area.pack_start(url_label, False, False, 10)

        self.url_entry = Gtk.Entry()
        self.url_entry.set_placeholder_text("https://api.openai.com/v1")
        content_area.pack_start(self.url_entry, False, False, 0)

        # Load existing settings if available
        self.load_settings()

        # Show all widgets
        self.show_all()

    def load_settings(self):
        """Load existing settings into the dialog"""
        # Get settings from app config
        config = self.parent.app.config_params.get('llm', {})

        # Set API key if available
        if 'default' in config and 'api_key' in config['default']:
            self.api_entry.set_text(config['default']['api_key'])

        # Set base URL if available
        if 'default' in config and 'base_url' in config['default']:
            self.url_entry.set_text(config['default']['base_url'])


def main():
    """Main entry point for the desktop application"""
    app = ManusApp()
    app.hold()  # Prevent garbage collection
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
