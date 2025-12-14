"""Tests for the Research CLI TUI."""

import pytest

from common.cli.tui.app import ResearchApp, COMMANDS


class TestResearchApp:
    """Test the main TUI application."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_app_starts(self):
        """Test that the app starts without errors."""
        app = ResearchApp()
        async with app.run_test() as pilot:
            # App should be running
            assert pilot.app.is_running

    @pytest.mark.asyncio(loop_scope="function")
    async def test_quit_with_q(self):
        """Test that pressing 'q' quits the app."""
        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.press("q")
            # App should have exited
            assert not pilot.app.is_running

    @pytest.mark.asyncio(loop_scope="function")
    async def test_has_command_provider(self):
        """Test that the app has our custom command provider registered."""
        from common.cli.tui.app import ResearchCommandProvider

        # Check that ResearchCommandProvider is in the COMMANDS set
        assert ResearchCommandProvider in ResearchApp.COMMANDS

    @pytest.mark.asyncio(loop_scope="function")
    async def test_home_screen_has_category_cards(self):
        """Test that the home screen has category cards."""
        from common.cli.tui.screens import HomeScreen

        app = ResearchApp()
        async with app.run_test() as pilot:
            # Wait for app to fully initialize
            await pilot.pause(delay=0.1)
            # Check that we have a HomeScreen
            screen = pilot.app.screen
            assert isinstance(screen, HomeScreen)
            # Check for category cards
            cards = screen.query(".category-card")
            assert len(cards) == 3  # Download, Preprocess, Analyze

    @pytest.mark.asyncio(loop_scope="function")
    async def test_home_screen_has_header_and_footer(self):
        """Test that the home screen has header and footer."""
        from textual.widgets import Header, Footer

        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause(delay=0.1)
            headers = pilot.app.screen.query(Header)
            footers = pilot.app.screen.query(Footer)
            assert len(headers) == 1
            assert len(footers) == 1

    @pytest.mark.asyncio(loop_scope="function")
    async def test_pressing_1_opens_download_category(self):
        """Test that pressing '1' opens the download category screen."""
        from common.cli.tui.screens.category import CategoryScreen

        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause(delay=0.1)
            initial_stack_size = len(pilot.app.screen_stack)
            await pilot.press("1")
            await pilot.pause(delay=0.1)
            # Should have CategoryScreen on stack (one more screen)
            assert len(pilot.app.screen_stack) > initial_stack_size

    @pytest.mark.asyncio(loop_scope="function")
    async def test_escape_goes_back(self):
        """Test that escape goes back from category screen."""
        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause(delay=0.1)
            initial_stack_size = len(pilot.app.screen_stack)
            await pilot.press("1")  # Go to download category
            await pilot.pause(delay=0.1)
            assert len(pilot.app.screen_stack) > initial_stack_size
            await pilot.press("escape")  # Go back
            await pilot.pause(delay=0.1)
            assert len(pilot.app.screen_stack) == initial_stack_size

    @pytest.mark.asyncio(loop_scope="function")
    async def test_arrow_key_navigation(self):
        """Test that arrow keys navigate between category cards."""
        from common.cli.tui.screens.home import CategoryCard

        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause(delay=0.1)
            # First card should be focused on mount
            focused = pilot.app.screen.focused
            assert isinstance(focused, CategoryCard)
            assert focused.category_id == "download"

            # Press right to go to preprocess
            await pilot.press("right")
            await pilot.pause(delay=0.05)
            focused = pilot.app.screen.focused
            assert isinstance(focused, CategoryCard)
            assert focused.category_id == "preprocess"

            # Press right to go to analyze
            await pilot.press("right")
            await pilot.pause(delay=0.05)
            focused = pilot.app.screen.focused
            assert isinstance(focused, CategoryCard)
            assert focused.category_id == "analyze"

            # Press right again to wrap to download
            await pilot.press("right")
            await pilot.pause(delay=0.05)
            focused = pilot.app.screen.focused
            assert isinstance(focused, CategoryCard)
            assert focused.category_id == "download"

            # Press left to wrap to analyze
            await pilot.press("left")
            await pilot.pause(delay=0.05)
            focused = pilot.app.screen.focused
            assert isinstance(focused, CategoryCard)
            assert focused.category_id == "analyze"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_enter_selects_focused_card(self):
        """Test that enter opens the focused category."""
        from common.cli.tui.screens.category import CategoryScreen

        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause(delay=0.1)
            initial_stack_size = len(pilot.app.screen_stack)
            # First card (download) should be focused
            await pilot.press("enter")
            await pilot.pause(delay=0.1)
            # Should have navigated to category screen
            assert len(pilot.app.screen_stack) > initial_stack_size
            assert isinstance(pilot.app.screen, CategoryScreen)
            assert pilot.app.screen.category_id == "download"


class TestCommands:
    """Test the command definitions."""

    def test_commands_list_not_empty(self):
        """Test that we have commands defined."""
        assert len(COMMANDS) > 0

    def test_commands_have_required_fields(self):
        """Test that each command has all required fields."""
        for cmd in COMMANDS:
            assert len(cmd) == 4
            name, description, module, args = cmd
            assert isinstance(name, str) and len(name) > 0
            assert isinstance(description, str) and len(description) > 0
            assert isinstance(module, str) and module.startswith("common.cli.")
            assert isinstance(args, list)

    def test_experiment_commands_exist(self):
        """Test that experiment commands are defined."""
        cmd_names = [cmd[0] for cmd in COMMANDS]
        assert "experiments list" in cmd_names
        assert "experiments summary" in cmd_names

    def test_infra_commands_exist(self):
        """Test that infrastructure commands are defined."""
        cmd_names = [cmd[0] for cmd in COMMANDS]
        assert "infra env" in cmd_names
        assert "infra lambda" in cmd_names

    def test_data_commands_exist(self):
        """Test that data commands are defined."""
        cmd_names = [cmd[0] for cmd in COMMANDS]
        assert any("data download" in name for name in cmd_names)
        assert any("data analyze" in name for name in cmd_names)

    def test_all_modules_are_valid(self):
        """Test that all command modules follow expected pattern."""
        valid_modules = [
            "common.cli.data",
            "common.cli.experiments",
            "common.cli.infra",
        ]
        for cmd in COMMANDS:
            module = cmd[2]
            assert module in valid_modules, f"Invalid module: {module}"


class TestWizardScreens:
    """Test wizard screen functionality."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_download_dataset_wizard_creates(self):
        """Test that download dataset wizard can be created."""
        from common.cli.tui.screens import DownloadDatasetScreen

        screen = DownloadDatasetScreen()
        assert screen.TITLE == "Download Dataset"
        assert screen.COMMAND_MODULE == "common.cli.data"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_wizard_base_get_command_args(self):
        """Test wizard base class get_command_args."""
        from common.cli.tui.screens import WizardScreen

        class TestWizard(WizardScreen):
            TITLE = "Test"
            COMMAND_MODULE = "test.module"

            def get_command_args(self):
                return ["arg1", "--flag", "value"]

        wizard = TestWizard()
        args = wizard.get_command_args()
        assert args == ["arg1", "--flag", "value"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_output_screen_creates(self):
        """Test that output screen can be created."""
        from common.cli.tui.screens import OutputScreen

        screen = OutputScreen("common.cli.infra", ["env"])
        assert screen.module == "common.cli.infra"
        assert screen.args == ["env"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_all_wizards_create(self):
        """Test that all wizard screens can be created."""
        from common.cli.tui.screens import (
            DownloadDatasetScreen,
            DownloadModelScreen,
            DownloadFinewebScreen,
            PretokenizeScreen,
            FinewebIndexScreen,
            FinewebExtractScreen,
            AnalyzeTokensScreen,
            QueryDomainsScreen,
            TrainTokenizerScreen,
        )

        wizards = [
            (DownloadDatasetScreen, "Download Dataset", "common.cli.data"),
            (DownloadModelScreen, "Download Model", "common.cli.data"),
            (DownloadFinewebScreen, "Download FineWeb Sample", "common.cli.data"),
            (PretokenizeScreen, "Pre-tokenize Dataset", "common.cli.data"),
            (FinewebIndexScreen, "Build FineWeb Index", "common.cli.data"),
            (FinewebExtractScreen, "Extract FineWeb Corpus", "common.cli.data"),
            (AnalyzeTokensScreen, "Analyze Tokens", "common.cli.data"),
            (QueryDomainsScreen, "Query Domain Index", "common.cli.data"),
            (TrainTokenizerScreen, "Train Tokenizer", "common.cli.data"),
        ]

        for wizard_class, expected_title, expected_module in wizards:
            screen = wizard_class()
            assert screen.TITLE == expected_title, f"{wizard_class.__name__} has wrong title"
            assert screen.COMMAND_MODULE == expected_module, f"{wizard_class.__name__} has wrong module"


class TestDatasetSelectWidget:
    """Test the DatasetSelect autocomplete widget."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_dataset_select_typing_filters_options(self):
        """Test that typing filters the options list."""
        from textual.app import App, ComposeResult
        from textual.widgets import Input, OptionList
        from common.cli.tui.widgets import DatasetSelect

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield DatasetSelect(id="ds")

        app = TestApp()
        async with app.run_test() as pilot:
            ds = pilot.app.query_one(DatasetSelect)
            # Mock some datasets
            ds._all_datasets = ["org/dataset-one", "org/dataset-two", "other/data"]

            # Type to filter
            input_widget = ds.query_one(Input)
            input_widget.value = "org"
            await pilot.pause(delay=0.05)

            # Check options are shown
            option_list = ds.query_one(OptionList)
            assert option_list.option_count == 2  # org/dataset-one, org/dataset-two

    @pytest.mark.asyncio(loop_scope="function")
    async def test_dataset_select_shows_all_on_focus(self):
        """Test that focusing the input shows all available datasets."""
        from textual.app import App, ComposeResult
        from textual.widgets import Input, OptionList
        from common.cli.tui.widgets import DatasetSelect

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield DatasetSelect(id="ds")

        app = TestApp()
        async with app.run_test() as pilot:
            ds = pilot.app.query_one(DatasetSelect)
            # Mock some datasets
            ds._all_datasets = ["org/dataset-one", "org/dataset-two", "other/data"]

            # Click on the widget to trigger on_click
            await pilot.click(DatasetSelect)
            await pilot.pause(delay=0.05)

            # Check options are shown
            option_list = ds.query_one(OptionList)
            assert "visible" in option_list.classes
            assert option_list.option_count == 3  # All datasets

    @pytest.mark.asyncio(loop_scope="function")
    async def test_dataset_select_selection_updates_value(self):
        """Test that selecting an option updates the input value."""
        from textual.app import App, ComposeResult
        from textual.widgets import Input, OptionList
        from common.cli.tui.widgets import DatasetSelect

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield DatasetSelect(id="ds")

        app = TestApp()
        async with app.run_test() as pilot:
            ds = pilot.app.query_one(DatasetSelect)
            ds._all_datasets = ["org/dataset-one", "org/dataset-two"]

            # Click to show options
            await pilot.click(DatasetSelect)
            await pilot.pause(delay=0.05)

            option_list = ds.query_one(OptionList)
            assert option_list.option_count == 2

            # Simulate selecting an option
            ds._select_dataset("org/dataset-two")
            await pilot.pause(delay=0.05)

            # Value should be updated
            assert ds.value == "org/dataset-two"
            input_widget = ds.query_one(Input)
            assert input_widget.value == "org/dataset-two"


class TestTokenizerSelectWidget:
    """Test the TokenizerSelect autocomplete widget."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_tokenizer_select_has_common_tokenizers(self):
        """Test that TokenizerSelect has common tokenizers predefined."""
        from textual.app import App, ComposeResult
        from textual.widgets import Input, OptionList
        from common.cli.tui.widgets import TokenizerSelect

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield TokenizerSelect(id="ts")

        app = TestApp()
        async with app.run_test() as pilot:
            ts = pilot.app.query_one(TokenizerSelect)

            # Click to show all options
            await pilot.click(TokenizerSelect)
            await pilot.pause(delay=0.05)

            option_list = ts.query_one(OptionList)
            # Should show common tokenizers (capped at 10)
            assert option_list.option_count <= 10
            assert option_list.option_count > 0
            assert "visible" in option_list.classes

    @pytest.mark.asyncio(loop_scope="function")
    async def test_tokenizer_select_typing_filters(self):
        """Test that typing filters the tokenizer options."""
        from textual.app import App, ComposeResult
        from textual.widgets import Input, OptionList
        from common.cli.tui.widgets import TokenizerSelect

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield TokenizerSelect(id="ts")

        app = TestApp()
        async with app.run_test() as pilot:
            ts = pilot.app.query_one(TokenizerSelect)

            # Type to filter
            input_widget = ts.query_one(Input)
            input_widget.value = "bert"
            await pilot.pause(delay=0.05)

            # Check options are filtered to bert tokenizers
            option_list = ts.query_one(OptionList)
            assert option_list.option_count > 0
            # All options should contain "bert"
            for i in range(option_list.option_count):
                opt = option_list.get_option_at_index(i)
                assert "bert" in opt.id.lower()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_tokenizer_select_selection_updates_value(self):
        """Test that selecting updates the input value."""
        from textual.app import App, ComposeResult
        from textual.widgets import Input, OptionList
        from common.cli.tui.widgets import TokenizerSelect

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield TokenizerSelect(id="ts")

        app = TestApp()
        async with app.run_test() as pilot:
            ts = pilot.app.query_one(TokenizerSelect)

            # Simulate selecting a tokenizer
            ts._select_tokenizer("gpt2-large")
            await pilot.pause(delay=0.05)

            # Value should be updated
            assert ts.value == "gpt2-large"
            input_widget = ts.query_one(Input)
            assert input_widget.value == "gpt2-large"


class TestFileSelectWidget:
    """Test the FileSelect autocomplete widget."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_file_select_creates(self):
        """Test that FileSelect widget can be created."""
        from textual.app import App, ComposeResult
        from textual.widgets import Input
        from common.cli.tui.widgets import FileSelect

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield FileSelect(
                    label="Test File",
                    extensions=[".jsonl"],
                    id="fs"
                )

        app = TestApp()
        async with app.run_test() as pilot:
            fs = pilot.app.query_one(FileSelect)
            assert fs is not None
            # Check input exists
            input_widget = fs.query_one(Input)
            assert input_widget is not None

    @pytest.mark.asyncio(loop_scope="function")
    async def test_file_select_get_value(self):
        """Test that FileSelect returns correct value."""
        from textual.app import App, ComposeResult
        from textual.widgets import Input
        from common.cli.tui.widgets import FileSelect

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield FileSelect(id="fs")

        app = TestApp()
        async with app.run_test() as pilot:
            fs = pilot.app.query_one(FileSelect)
            input_widget = fs.query_one(Input)

            # Set a value
            input_widget.value = "/path/to/file.jsonl"
            await pilot.pause(delay=0.05)

            # get_value should return it
            assert fs.get_value() == "/path/to/file.jsonl"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_file_select_path_completions(self):
        """Test path completion helper function."""
        from pathlib import Path
        from common.cli.tui.widgets.file_select import get_path_completions

        # Get completions for current directory (should find something)
        completions = get_path_completions("", base_dir=Path.cwd())
        # Should return some results (directories or files in cwd)
        assert isinstance(completions, list)
        # All results should be strings
        for c in completions:
            assert isinstance(c, str)
