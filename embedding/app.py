import streamlit as st
from query_data import perform_query


def main():
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF")
    st.write("The PDF document is a book about Steve Jobs. \
             Try to ask a question about Steve Jobs.")
    query_text = st.text_input("Ask your question:")

    if query_text:
        if query_text:  # Check if query_text is not empty
            response_text, sources = perform_query(query_text)
            if response_text is None:
                st.write("Sorry, I could not find an answer to your question.")
                st.write("Sources:")
                for source in sources:
                    st.write(source)
                return
            st.write("Answer:")
            st.write(response_text)  # Display the response

            st.write("------")
            st.write("Sources:")
            for source in sources:
                st.write(source)


if __name__ == "__main__":
    main()
