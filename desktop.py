#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenManus Desktop Application
A Windows desktop application for the OpenManus agent using GTK.
"""

# Platform detection
# import platform
import os
import sys
import threading
import asyncio
import ctypes
from typing import Optional, Dict, Any, List
# os.environ['PATH'] = r'C:\tools\msys64\mingw64\bin;' + os.environ['PATH']
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
        window = MainWindow(self)
        window.show_all()
        window.present()  # Force focus

    def initialize_agent(self):
        """Initialize the Manus agent"""
        self.agent = Manus()
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

        # Initialize UI components
        self.build_ui()

        # Connect signals
        self.connect("delete-event", self.on_close)

    def build_ui(self):
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

        # Model selection
        model_label = Gtk.Label(label="Model:")
        model_label.set_halign(Gtk.Align.START)
        sidebar_box.pack_start(model_label, False, False, 0)

        model_combo = Gtk.ComboBoxText()
        models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"]
        for model in models:
            model_combo.append_text(model)
        model_combo.set_active(0)
        model_combo.connect("changed", self.on_model_changed)
        sidebar_box.pack_start(model_combo, False, False, 5)

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
        temp_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=temp_adj)
        temp_scale.set_digits(1)
        temp_scale.set_value_pos(Gtk.PositionType.RIGHT)
        temp_scale.connect("value-changed", self.on_temperature_changed)
        sidebar_box.pack_start(temp_scale, False, False, 0)

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
        tokens_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=tokens_adj)
        tokens_scale.set_digits(0)
        tokens_scale.set_value_pos(Gtk.PositionType.RIGHT)
        tokens_scale.connect("value-changed", self.on_tokens_changed)
        sidebar_box.pack_start(tokens_scale, False, False, 0)

        # Tool selection
        tools_label = Gtk.Label(label="Available Tools:")
        tools_label.set_halign(Gtk.Align.START)
        sidebar_box.pack_start(tools_label, False, False, 10)

        # Tool checkboxes
        tools_frame = Gtk.Frame()
        tools_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        tools_frame.add(tools_box)

        tool_names = ["Python Execute", "Web Search", "Browser Use", "File Saver"]
        for tool in tool_names:
            checkbox = Gtk.CheckButton.new_with_label(tool)
            checkbox.set_active(True)
            checkbox.connect("toggled", self.on_tool_toggled, tool)
            tools_box.pack_start(checkbox, False, False, 0)

        sidebar_box.pack_start(tools_frame, False, False, 0)

        # Spacer
        spacer = Gtk.Label()
        sidebar_box.pack_start(spacer, True, True, 0)

        # Run button
        run_button = Gtk.Button.new_with_label("Run Agent")
        run_button.connect("clicked", self.on_run_clicked)
        sidebar_box.pack_end(run_button, False, False, 0)

        return sidebar_box

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
        self.prompt_view = Gtk.TextView.new_with_buffer(self.prompt_buffer)
        self.prompt_view.set_wrap_mode(Gtk.WrapMode.WORD)
        prompt_scroll.add(self.prompt_view)

        content_box.pack_start(prompt_scroll, False, False, 0)

        # Submit button
        submit_button = Gtk.Button.new_with_label("Submit")
        submit_button.connect("clicked", self.on_submit_clicked)
        content_box.pack_start(submit_button, False, False, 0)

        # Response display area
        response_label = Gtk.Label(label="Response:")
        response_label.set_halign(Gtk.Align.START)
        content_box.pack_start(response_label, False, False, 10)

        response_scroll = Gtk.ScrolledWindow()
        response_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        self.response_buffer = Gtk.TextBuffer()
        self.response_view = Gtk.TextView.new_with_buffer(self.response_buffer)
        self.response_view.set_wrap_mode(Gtk.WrapMode.WORD)
        self.response_view.set_editable(False)
        response_scroll.add(self.response_view)

        content_box.pack_start(response_scroll, True, True, 0)

        # Progress indicator
        self.progress_bar = Gtk.ProgressBar()
        self.progress_bar.set_pulse_step(0.1)
        content_box.pack_start(self.progress_bar, False, False, 0)

        return content_box

    def on_model_changed(self, combo):
        """Handle model selection change"""
        model = combo.get_active_text()
        logger.info(f"Model changed to: {model}")
        # Update config
        if self.app.config_params.get('llm') is None:
            self.app.config_params['llm'] = {}
        self.app.config_params['llm']['model'] = model

    def on_temperature_changed(self, scale):
        """Handle temperature setting change"""
        temp = scale.get_value()
        logger.info(f"Temperature changed to: {temp}")
        # Update config
        if self.app.config_params.get('llm') is None:
            self.app.config_params['llm'] = {}
        self.app.config_params['llm']['temperature'] = temp

    def on_tokens_changed(self, scale):
        """Handle max tokens setting change"""
        tokens = int(scale.get_value())
        logger.info(f"Max tokens changed to: {tokens}")
        # Update config
        if self.app.config_params.get('llm') is None:
            self.app.config_params['llm'] = {}
        self.app.config_params['llm']['max_tokens'] = tokens

    def on_tool_toggled(self, checkbox, tool_name):
        """Handle tool checkbox toggle"""
        active = checkbox.get_active()
        logger.info(f"Tool '{tool_name}' {'enabled' if active else 'disabled'}")
        # Update config (to be implemented)

    def on_settings_clicked(self, button):
        """Handle settings button click"""
        dialog = SettingsDialog(self)
        dialog.run()
        dialog.destroy()

    def on_run_clicked(self, button):
        """Handle run button click"""
        # Initialize agent if not already initialized
        if self.app.agent is None:
            self.app.initialize_agent()
            self.status_bar.push(self.status_context, "Agent initialized")

    def on_submit_clicked(self, button):
        """Handle submit button click"""
        # Get prompt from text view
        start_iter = self.prompt_buffer.get_start_iter()
        end_iter = self.prompt_buffer.get_end_iter()
        prompt = self.prompt_buffer.get_text(start_iter, end_iter, True)

        if not prompt.strip():
            self.show_error_dialog("Please enter a prompt.")
            return

        # Clear response
        self.response_buffer.set_text("")

        # Start progress indicator
        self.start_progress_pulse()
        self.status_bar.push(self.status_context, "Processing request...")

        # Initialize agent if not already initialized
        if self.app.agent is None:
            self.app.initialize_agent()

        # Run agent in background
        self.app.run_background_task(
            self.app.agent.run,
            self.on_agent_response,
            prompt
        )

    def on_agent_response(self, result, error=None):
        """Handle agent response"""
        # Stop progress indicator
        self.stop_progress_pulse()

        if error:
            self.show_error_dialog(f"Error: {error}")
            self.status_bar.push(self.status_context, "Error processing request")
            return

        # Update response display
        self.response_buffer.set_text("Request processing completed.")
        self.status_bar.push(self.status_context, "Request processing completed")

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
        if 'api_key' in config:
            self.api_entry.set_text(config['api_key'])

        # Set base URL if available
        if 'base_url' in config:
            self.url_entry.set_text(config['base_url'])


def main():
    """Main entry point for the desktop application"""
    app = ManusApp()
    app.hold()  # Prevent garbage collection
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
