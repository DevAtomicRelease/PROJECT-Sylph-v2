import os
import glob
import logging

logger = logging.getLogger("sylph.tools.file")

def list_files(directory: str) -> list[str]:
    """
    List files and directories in a given directory path.

    Args:
        directory: Absolute path to the directory.

    Returns:
        List of file and directory names.
    """
    try:
        # Expand user path (e.g. ~ or ~/Desktop)
        expanded_dir = os.path.abspath(os.path.expanduser(directory))
        logger.info("Listing files in directory: %s", expanded_dir)
        
        if not os.path.exists(expanded_dir):
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not os.path.isdir(expanded_dir):
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        items = os.listdir(expanded_dir)
        logger.info("Found %d items in %s", len(items), expanded_dir)
        return items
    except Exception as e:
        logger.error("Failed to list files: %s", e)
        raise e

def read_file(file_path: str) -> str:
    """
    Read text from a local file. Supports text files and PDFs (via pypdf fallback).

    Args:
        file_path: Path to the file.

    Returns:
        The content/text of the file.
    """
    try:
        expanded_path = os.path.abspath(os.path.expanduser(file_path))
        logger.info("Reading file: %s", expanded_path)

        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check file extension
        ext = os.path.splitext(expanded_path)[1].lower()

        # Handle PDF files
        if ext == ".pdf":
            try:
                # Try pypdf first since it is pure Python and very reliable
                import pypdf
                reader = pypdf.PdfReader(expanded_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                logger.info("Successfully parsed PDF via pypdf (%d chars)", len(text))
                return text
            except Exception as e:
                logger.warning("pypdf parsing failed, attempting Tika fallback: %s", e)
                try:
                    # Fallback to Apache Tika via tika-python
                    from tika import parser
                    parsed = parser.from_file(expanded_path)
                    text = parsed.get("content", "")
                    if text:
                        return text
                except Exception as t_err:
                    logger.error("All PDF parsers failed: %s", t_err)
                    raise RuntimeError("Unable to parse PDF file. Ensure dependencies are installed.")

        # Handle plain text files
        try:
            with open(expanded_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            logger.info("Read plain text file: %d characters", len(content))
            return content
        except UnicodeDecodeError:
            raise ValueError("File is not a readable plain text file (unicode decode failed)")

    except Exception as e:
        logger.error("Failed to read file: %s", e)
        raise e

def search_files(query: str, directory: str) -> list[str]:
    """
    Search for files containing query in their name or content.

    Args:
        query: The query text to search.
        directory: The directory to search inside.

    Returns:
        A list of matching file paths.
    """
    try:
        expanded_dir = os.path.abspath(os.path.expanduser(directory))
        logger.info("Searching for '%s' in directory '%s'", query, expanded_dir)

        if not os.path.exists(expanded_dir):
            raise FileNotFoundError(f"Directory not found: {directory}")

        matches = []
        # Walk directory recursively
        for root, dirs, files in os.walk(expanded_dir):
            for file in files:
                # Match file name
                if query.lower() in file.lower():
                    matches.append(os.path.join(root, file))
                    continue

                # Match file content for text files
                ext = os.path.splitext(file)[1].lower()
                if ext in [".txt", ".md", ".py", ".js", ".json", ".html", ".css"]:
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            if query.lower() in f.read().lower():
                                matches.append(full_path)
                    except Exception:
                        pass

        logger.info("Found %d search matches", len(matches))
        return matches[:20]  # Cap at 20 matches
    except Exception as e:
        logger.error("Search failed: %s", e)
        raise e
