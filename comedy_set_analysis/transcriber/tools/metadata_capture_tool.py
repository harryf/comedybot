import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_tool import BaseTool
from pydantic import Field, PrivateAttr
import json
from datetime import datetime, timedelta
import logging
import argparse
import objc
import threading
from multiprocessing import Process
import time

from Cocoa import (
    NSApplication, NSTextField, NSButton, NSWindow, NSDatePicker,
    NSView, NSRect, NSPoint, NSSize, NSBackingStoreBuffered,
    NSTitledWindowMask, NSClosableWindowMask, NSApp, NSImage,
    NSTextView, NSScrollView, NSBezelBorder, NSDatePickerElementFlagYearMonthDay,
    NSDatePickerElementFlagHourMinute, NSDatePickerModeSingle,
    NSTextFieldSquareBezel, NSApplicationActivationPolicyRegular,
    NSObject, NSTextFieldCell
)
from Foundation import NSDate, NSDateFormatter, NSString, NSURL
from AppKit import (
    NSViewWidthSizable, NSFont, NSPasteboard, 
    NSEventModifierFlagCommand, NSEventModifierFlagControl,
    NSPasteboardTypeString
)

logger = logging.getLogger(__name__)

# Create custom classes
class CustomTextField(NSTextField):
    def initWithFrame_(self, frame):
        self = objc.super(CustomTextField, self).initWithFrame_(frame)
        if self:
            cell = NSTextFieldCell.alloc().init()
            cell.setEditable_(True)
            cell.setSelectable_(True)
            cell.setStringValue_("")
            cell.setBezeled_(True)
            cell.setDrawsBackground_(True)
            self.setCell_(cell)
            self.setEditable_(True)
            self.setSelectable_(True)
            self.setStringValue_("")
            self.setBezeled_(True)
            self.setDrawsBackground_(True)
        return self

    def textDidBeginEditing_(self, notification):
        """Handle text editing start"""
        editor = self.currentEditor()
        if editor:
            editor.setSelectedRange_((0, len(self.stringValue())))
        objc.super(CustomTextField, self).textDidBeginEditing_(notification)

    def mouseDown_(self, event):
        """Handle mouse clicks"""
        if self.isEditable():
            self.window().makeFirstResponder_(self)
            editor = self.currentEditor()
            if editor:
                editor.setSelectedRange_((0, len(self.stringValue())))
        objc.super(CustomTextField, self).mouseDown_(event)

    def keyDown_(self, event):
        # Handle keyboard shortcuts for both Command and Control
        modifiers = event.modifierFlags()
        if modifiers & (NSEventModifierFlagCommand | NSEventModifierFlagControl):
            character = event.characters()
            if character == 'v':  # Paste
                self.paste_(self)
                return True
            elif character == 'c':  # Copy
                self.copy_(self)
                return True
            elif character == 'x':  # Cut
                self.cut_(self)
                return True
        return objc.super(CustomTextField, self).keyDown_(event)

    def copy_(self, sender):
        pasteboard = NSPasteboard.generalPasteboard()
        pasteboard.clearContents()
        pasteboard.setString_forType_(self.stringValue(), NSPasteboardTypeString)

    def cut_(self, sender):
        self.copy_(sender)
        self.setStringValue_("")

    def paste_(self, sender):
        pasteboard = NSPasteboard.generalPasteboard()
        if pasteboard.types().containsObject_(NSPasteboardTypeString):
            pasted_text = pasteboard.stringForType_(NSPasteboardTypeString)
            if pasted_text:
                self.setStringValue_(pasted_text)

class GMapsTextField(CustomTextField):
    def initWithFrame_(self, frame):
        self = objc.super(GMapsTextField, self).initWithFrame_(frame)
        if self:
            cell = NSTextFieldCell.alloc().init()
            cell.setEditable_(True)
            cell.setSelectable_(True)
            cell.setPlaceholderString_("Paste a Google Maps link here")
            cell.setStringValue_("")
            cell.setBezeled_(True)
            cell.setDrawsBackground_(True)
            self.setCell_(cell)
            self.setBezelStyle_(NSTextFieldSquareBezel)
            self.setEditable_(True)
            self.setSelectable_(True)
            self.setStringValue_("")
            self.setBezeled_(True)
            self.setDrawsBackground_(True)
        return self

    def clean_maps_url(self, url):
        """Clean and validate Google Maps URL"""
        import re
        maps_patterns = [
            r'https?://(?:www\.)?google\.com/maps\b.*',
            r'https?://(?:www\.)?maps\.google\.com\b.*',
            r'https?://(?:www\.)?goo\.gl/maps\b.*',
            r'https?://(?:www\.)?maps\.app\.goo\.gl/.*',
            r'https?://maps\.app\.goo\.gl/.*'
        ]
        url = url.strip()
        for pattern in maps_patterns:
            if re.match(pattern, url):
                return url
        return None

    def getStringValue(self):
        """Override to return cleaned URL or None"""
        raw_value = self.stringValue()
        if not raw_value:
            return None
        return self.clean_maps_url(raw_value)

class WindowDelegate(NSObject):
    def windowWillClose_(self, notification):
        NSApp().stopModal()

class MetadataCaptureWindow:
    def __init__(self, tool, file_name, file_date_time, output_directory):
        self.tool = tool
        self.app = NSApplication.sharedApplication()
        self.output_directory = output_directory
        self.metadata_captured = False
        
        # Create window delegate
        self.delegate = WindowDelegate.alloc().init()
        
        # Load application icon
        app_icon_path = os.path.join(os.path.dirname(__file__), "app_icon.png")
        if os.path.exists(app_icon_path):
            app_icon = NSImage.alloc().initWithContentsOfFile_(app_icon_path)
            self.app.setApplicationIconImage_(app_icon)

        # Create window
        window_style = NSTitledWindowMask | NSClosableWindowMask
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSRect(NSPoint(200, 500), NSSize(500, 600)),
            window_style,
            NSBackingStoreBuffered,
            False
        )
        self.window.setTitle_("Metadata Capture Tool")
        
        # Load window icon
        window_icon_path = os.path.join(os.path.dirname(__file__), "window_icon.png")
        if os.path.exists(window_icon_path):
            window_icon = NSImage.alloc().initWithContentsOfFile_(window_icon_path)
            # Set window's miniaturize (minimize) image
            self.window.setMiniwindowImage_(window_icon)
            # Set window's title bar icon
            self.window.setRepresentedFilename_(window_icon_path)
            self.window.setRepresentedURL_(NSURL.fileURLWithPath_(window_icon_path))
        
        # Set the window delegate
        self.window.setDelegate_(self.delegate)
        
        # Create content view
        self.content_view = NSView.alloc().initWithFrame_(
            NSRect(NSPoint(0, 0), NSSize(500, 600))
        )
        
        # Load existing metadata if it exists
        self.existing_metadata = None
        metadata_path = os.path.join(output_directory, 'metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.existing_metadata = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse existing metadata.json in {output_directory}")

        self.text_fields = {}
        self.create_form_elements(file_name, file_date_time)
        
        # Set content view and show window
        self.window.setContentView_(self.content_view)
        
        # Handle window close event
        self.window.setReleasedWhenClosed_(False)

        # Enable copy/paste menu items for the window
        self.window.setAcceptsMouseMovedEvents_(True)
        self.window.makeFirstResponder_(None)

    def create_form_elements(self, file_name, file_date_time):
        y_position = 550
        
        # Title
        self.add_label("Enter Show Details", y_position, font_size=18)
        
        # File name and date
        y_position -= 30
        self.add_label(f"File: {file_name} | Date: {file_date_time}", y_position, font_size=12)
        
        # Comedian (Required)
        y_position -= 40
        self.add_label("Comedian *:", y_position)
        comedian_field = self.add_text_field(y_position, placeholder="Enter comedian name")
        comedian_field.setStringValue_("Harry Fücks")  # Set default value
        if self.existing_metadata and self.existing_metadata.get("comedian"):
            comedian_field.setStringValue_(self.existing_metadata["comedian"])
        self.text_fields["comedian"] = comedian_field
        
        # Show Name (Required)
        y_position -= 40
        self.add_label("Name of Show *:", y_position)
        show_name_field = self.add_text_field(y_position, placeholder="Enter show name")
        if self.existing_metadata and self.existing_metadata.get("name_of_show"):
            show_name_field.setStringValue_(self.existing_metadata["name_of_show"])
        self.text_fields["name_of_show"] = show_name_field
        
        # Date and Time Picker
        y_position -= 40
        self.add_label("Date of Show *:", y_position)
        
        # Create date picker
        date_picker = NSDatePicker.alloc().initWithFrame_(
            NSRect(NSPoint(170, y_position), NSSize(200, 24))
        )
        date_picker.setDatePickerElements_(
            NSDatePickerElementFlagYearMonthDay | NSDatePickerElementFlagHourMinute
        )
        date_picker.setDatePickerMode_(NSDatePickerModeSingle)
        
        # Set date from existing metadata or default to yesterday at 20:00
        if self.existing_metadata and self.existing_metadata.get("date_of_show"):
            try:
                date_formatter = NSDateFormatter.alloc().init()
                date_formatter.setDateFormat_("dd MMM yyyy, HH:mm")
                date = date_formatter.dateFromString_(self.existing_metadata["date_of_show"])
                if date:
                    date_picker.setDateValue_(date)
            except Exception as e:
                logger.error(f"Failed to parse existing date: {e}")
                # Fall back to default date
                yesterday = datetime.now() - timedelta(days=1)
                yesterday = yesterday.replace(hour=20, minute=0, second=0, microsecond=0)
                date_picker.setDateValue_(NSDate.dateWithTimeIntervalSince1970_(
                    yesterday.timestamp()
                ))
        else:
            yesterday = datetime.now() - timedelta(days=1)
            yesterday = yesterday.replace(hour=20, minute=0, second=0, microsecond=0)
            date_picker.setDateValue_(NSDate.dateWithTimeIntervalSince1970_(
                yesterday.timestamp()
            ))
        
        self.content_view.addSubview_(date_picker)
        self.text_fields["date_picker"] = date_picker
        
        # Venue Name
        y_position -= 40
        self.add_label("Name of Venue:", y_position)
        venue_field = self.add_text_field(y_position, placeholder="Enter venue name")
        if self.existing_metadata and self.existing_metadata.get("name_of_venue"):
            venue_field.setStringValue_(self.existing_metadata["name_of_venue"])
        self.text_fields["name_of_venue"] = venue_field
        
        # Venue Google Maps Link
        y_position -= 40
        self.add_label("Venue GMaps Link:", y_position)
        maps_field = GMapsTextField.alloc().initWithFrame_(
            NSRect(NSPoint(170, y_position), NSSize(300, 24))
        )
        if self.existing_metadata and self.existing_metadata.get("link_to_venue_on_google_maps"):
            maps_field.setStringValue_(self.existing_metadata["link_to_venue_on_google_maps"])
        self.content_view.addSubview_(maps_field)
        self.text_fields["link_to_venue_on_google_maps"] = maps_field
        
        # Payment Amount
        y_position -= 40
        self.add_label("Payment Amount:", y_position)
        payment_field = self.add_text_field(y_position, width=100, placeholder="0")
        # Set default value to 0 or existing value
        default_payment = "0"
        if self.existing_metadata and self.existing_metadata.get("payment", {}).get("amount") is not None:
            default_payment = str(self.existing_metadata["payment"]["amount"])
        payment_field.setStringValue_(default_payment)
        self.text_fields["payment_amount"] = payment_field
        
        # Currency
        self.add_label("Currency:", y_position, x=280)
        currency_field = self.add_text_field(y_position, x=350, width=70, placeholder="CHF")
        # Set default value to CHF or existing value
        default_currency = "CHF"
        if self.existing_metadata and self.existing_metadata.get("payment", {}).get("currency"):
            default_currency = self.existing_metadata["payment"]["currency"]
        currency_field.setStringValue_(default_currency)
        self.text_fields["currency"] = currency_field
        
        # Notes (Text Area)
        y_position -= 40
        self.add_label("Notes:", y_position)
        
        # Create scroll view for notes
        scroll_view = NSScrollView.alloc().initWithFrame_(
            NSRect(NSPoint(170, y_position - 120), NSSize(300, 150))
        )
        scroll_view.setBorderType_(NSBezelBorder)
        scroll_view.setHasVerticalScroller_(True)
        
        # Create text view for notes
        text_view = NSTextView.alloc().initWithFrame_(
            NSRect(NSPoint(0, 0), NSSize(300, 150))
        )
        text_view.setMinSize_(NSSize(300, 150))
        text_view.setMaxSize_(NSSize(float("inf"), float("inf")))
        text_view.setVerticallyResizable_(True)
        text_view.setHorizontallyResizable_(False)
        text_view.setAutoresizingMask_(NSViewWidthSizable)
        
        if self.existing_metadata and self.existing_metadata.get("notes"):
            text_view.setString_(self.existing_metadata["notes"])
        
        scroll_view.setDocumentView_(text_view)
        self.content_view.addSubview_(scroll_view)
        self.text_fields["notes"] = text_view
        
        # Analysis Results (Read-only)
        y_position = 100  # Position above the submit/cancel buttons
        self.add_label("Analysis Results:", y_position, font_size=12)
        
        y_position -= 25
        length_label = self.add_label("Length of Set:", y_position, x=20, font_size=12)
        length_value = self.add_label(
            str((self.existing_metadata or {}).get("length_of_set", "Not analyzed yet") or "Not analyzed yet"), 
            y_position, x=170, font_size=12
        )
        
        y_position -= 25
        lpm_label = self.add_label("Laughs per Minute:", y_position, x=20, font_size=12)
        lpm_value = self.add_label(
            str((self.existing_metadata or {}).get("laughs_per_minute", "Not analyzed yet") or "Not analyzed yet"), 
            y_position, x=170, font_size=12
        )
        
        # Store analysis fields for updating
        self.text_fields["length_of_set"] = length_value
        self.text_fields["laughs_per_minute"] = lpm_value
        
        # Submit Button
        submit_button = NSButton.alloc().initWithFrame_(
            NSRect(NSPoint(200, 20), NSSize(100, 30))
        )
        submit_button.setTitle_("Submit")
        submit_button.setTarget_(self)
        submit_button.setAction_("submit:")
        self.content_view.addSubview_(submit_button)
        
        # Cancel Button
        cancel_button = NSButton.alloc().initWithFrame_(
            NSRect(NSPoint(310, 20), NSSize(100, 30))
        )
        cancel_button.setTitle_("Cancel")
        cancel_button.setTarget_(self)
        cancel_button.setAction_("cancel:")
        self.content_view.addSubview_(cancel_button)

    def add_label(self, text, y_position, x=10, font_size=14):
        label = CustomTextField.alloc().initWithFrame_(
            NSRect(NSPoint(x, y_position), NSSize(480, 24))
        )
        label.setStringValue_(text)
        label.setBezeled_(False)
        
        # Create and configure cell
        cell = NSTextFieldCell.alloc().init()
        cell.setEditable_(False)
        cell.setSelectable_(True)
        cell.setStringValue_(text)
        label.setCell_(cell)
        
        label.setDrawsBackground_(False)
        label.setFont_(NSFont.systemFontOfSize_(font_size))
        self.content_view.addSubview_(label)
        return label

    def add_text_field(self, y_position, x=170, width=200, placeholder=""):
        field = CustomTextField.alloc().initWithFrame_(
            NSRect(NSPoint(x, y_position), NSSize(width, 24))
        )
        field.setBezelStyle_(NSTextFieldSquareBezel)
        field.setStringValue_("")
        
        # Create and configure cell
        cell = NSTextFieldCell.alloc().init()
        cell.setEditable_(True)
        cell.setSelectable_(True)
        cell.setPlaceholderString_(placeholder)
        cell.setStringValue_("")
        field.setCell_(cell)
        
        self.content_view.addSubview_(field)
        return field

    def submit_(self, sender):
        # Validate required fields
        comedian = self.text_fields["comedian"].stringValue()
        if not comedian:
            self.show_error("Comedian name is required")
            return
            
        show_name = self.text_fields["name_of_show"].stringValue()
        if not show_name:
            self.show_error("Show name is required")
            return
            
        # Get and validate Google Maps URL
        maps_field = self.text_fields["link_to_venue_on_google_maps"]
        maps_link = maps_field.getStringValue()  # This will return None if invalid
        if maps_field.stringValue() and not maps_link:
            self.show_error("Please enter a valid Google Maps URL or leave the field empty")
            return
        
        # Format date
        date_formatter = NSDateFormatter.alloc().init()
        date_formatter.setDateFormat_("dd MMM yyyy, HH:mm")
        date_str = date_formatter.stringFromDate_(
            self.text_fields["date_picker"].dateValue()
        )
        
        # Get payment details
        try:
            payment_amount = self.text_fields["payment_amount"].stringValue()
            payment_amount = float(payment_amount) if payment_amount else None
        except ValueError:
            payment_amount = None
            
        currency = self.text_fields["currency"].stringValue() or None
        
        # Construct JSON data
        json_data = {
            "audio_file": os.path.basename(self.tool.audio_file_path),
            "comedian": comedian,
            "name_of_show": show_name,
            "date_of_show": date_str,
            "name_of_venue": self.text_fields["name_of_venue"].stringValue() or None,
            "link_to_venue_on_google_maps": maps_link,  # This will be None if invalid or empty
            "notes": self.text_fields["notes"].string() or None,
            "payment": {
                "amount": payment_amount,
                "currency": currency
            },
            # Store the segments list with the initial audio file
            "segments": [os.path.basename(self.tool.audio_file_path)],
            # Preserve existing analysis values if they exist
            "length_of_set": (self.existing_metadata or {}).get("length_of_set", None),
            "laughs_per_minute": (self.existing_metadata or {}).get("laughs_per_minute", None)
        }
        
        # Write to metadata.json
        metadata_path = os.path.join(self.output_directory, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(json_data, f, indent=4)
            
        self.metadata_captured = True
        
        # Just stop modal and close window
        NSApp().stopModal()
        self.window.close()

    def cancel_(self, sender):
        # Create a cancel signal file
        cancel_path = os.path.join(self.output_directory, '.metadata_cancelled')
        with open(cancel_path, 'w') as f:
            f.write('cancelled')
            
        # Just stop modal and close window
        NSApp().stopModal()
        self.window.close()
        
    def show_error(self, message):
        from AppKit import NSAlert, NSWarningAlertStyle
        alert = NSAlert.alloc().init()
        alert.setMessageText_("Error")
        alert.setInformativeText_(message)
        alert.setAlertStyle_(NSWarningAlertStyle)
        
        # Set alert icon
        app_icon_path = os.path.join(os.path.dirname(__file__), "app_icon.png")
        if os.path.exists(app_icon_path):
            icon = NSImage.alloc().initWithContentsOfFile_(app_icon_path)
            alert.setIcon_(icon)
            
        alert.runModal()

class MetadataCaptureTool(BaseTool):
    """
    Launches a GUI to collect metadata about a comedy show and saves it as metadata.json in the specified directory.
    """
    audio_file_path: str = Field(
        ..., 
        description="Path to the audio file being processed"
    )
    output_directory: str = Field(
        ..., 
        description="Directory where metadata.json should be saved"
    )

    def _run_gui(self):
        """Run the GUI in a separate process"""
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyRegular)

        # Set application icon
        app_icon_path = os.path.join(os.path.dirname(__file__), "app_icon.png")
        if os.path.exists(app_icon_path):
            app_icon = NSImage.alloc().initWithContentsOfFile_(app_icon_path)
            app.setApplicationIconImage_(app_icon)

        file_name = os.path.basename(self.audio_file_path)
        file_date_time = datetime.fromtimestamp(
            os.path.getmtime(self.audio_file_path)
        ).strftime('%Y-%m-%d %H:%M:%S')

        # Create and show the window
        form = MetadataCaptureWindow(self, file_name, file_date_time, self.output_directory)
        
        if form.window is not None:
            form.window.center()
            form.window.makeKeyAndOrderFront_(None)
            app.activateIgnoringOtherApps_(True)
            NSApp().runModalForWindow_(form.window)
        else:
            logger.error("Failed to initialize metadata capture window")

    def run(self):
        """
        Launches the metadata capture GUI in a separate process and waits for metadata.json to be created.
        Returns True if metadata was captured successfully, False if cancelled or timed out.
        """
        metadata_path = os.path.join(self.output_directory, 'metadata.json')
        cancel_path = os.path.join(self.output_directory, '.metadata_cancelled')
        backup_path = metadata_path + '.bak'
        
        # Clean up any existing cancel file
        if os.path.exists(cancel_path):
            os.remove(cancel_path)
        
        # If metadata.json already exists, rename it as backup
        if os.path.exists(metadata_path):
            os.rename(metadata_path, backup_path)
        
        # Start GUI in separate process
        gui_process = Process(target=self._run_gui)
        gui_process.start()

        # Wait for either metadata file or cancel signal
        timeout = 300  # 5 minutes in seconds
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if os.path.exists(metadata_path):
                # Give a small delay to ensure file is fully written
                time.sleep(0.5)
                gui_process.terminate()
                # Clean up cancel file if it exists
                if os.path.exists(cancel_path):
                    os.remove(cancel_path)
                return True
                
            if os.path.exists(cancel_path):
                # User cancelled
                gui_process.terminate()
                # Clean up cancel file
                os.remove(cancel_path)
                # Restore backup if it exists
                if os.path.exists(backup_path):
                    os.rename(backup_path, metadata_path)
                logger.info("Metadata capture was cancelled by user")
                return False
                
            time.sleep(1)  # Check every second

        # If we get here, we timed out
        gui_process.terminate()
        
        # Restore backup if it exists
        if os.path.exists(backup_path):
            os.rename(backup_path, metadata_path)
            
        logger.error("Metadata capture timed out")
        return False

    def cleanup(self):
        """Explicitly close the GUI window and cleanup resources"""
        if hasattr(self, '_form') and self._form and self._form.window:
            self._form.window.close()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Capture metadata for a comedy show audio file.")
    parser.add_argument('-a', '--audio_file_path', type=str, required=True, help='Path to the audio file being processed')
    parser.add_argument('-o', '--output_directory', type=str, required=True, help='Directory where metadata.json should be saved')

    # Parse arguments
    args = parser.parse_args()

    # Test the tool with command line arguments
    tool = MetadataCaptureTool(
        audio_file_path=args.audio_file_path,
        output_directory=args.output_directory
    )
    result = tool.run()
    print(f"Metadata captured: {result}") 