import pytest
from main import init_keys, make_file_upload, WorkFlow

class TestMainFunctionality:

    def test_init_keys(self):
        init_keys()
        assert "messages" in st.session_state
        assert "file_uploaded" in st.session_state

    def test_make_file_upload(self):
        make_file_upload()
        assert st.session_state.file_uploaded is True

    def test_workflow_execution(self):
        model = WorkFlow()
        result = model.execute_steps('Test input')
        assert result is not None

# Edge case tests
    def test_invalid_link(self):
        st.session_state.file_uploaded = False
        paper_link = 'invalid_link'
        with pytest.raises(Exception):
            # Assuming the function raises an exception on invalid link
            ArxivLoader(query=paper_link)

# Additional edge cases can be continued following similar structure.