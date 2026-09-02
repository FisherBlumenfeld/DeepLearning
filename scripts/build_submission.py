# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jinja2>=3.1.0",
#     "pygments>=2.17.0",
#     "markdown>=3.5",
#     "weasyprint>=60.0",
# ]
# ///
"""Builds a self-contained HTML/PDF submission from homework source files and vector SVG artifacts.

Usage:
    uv run scripts/build_submission.py <hw_name> [--pdf] [--output-dir <dir>] [--output <output_path>]
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import List, Tuple

import jinja2
import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import (
    TextLexer,
    get_lexer_by_name,
    get_lexer_for_filename,
)
from pygments.util import ClassNotFound


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

IGNORED_DIRS = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "checkpoints",
    "node_modules",
}

IGNORED_FILES = {
    ".DS_Store",
    "uv.lock",
    ".python-version",
    ".gitignore",
}

IGNORED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".ps",
    ".ps~",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".mp4",
    ".mov",
    ".ocdbt",
}

# Module priority within the package / project
CORE_MODULE_ORDER = [
    "pyproject.toml",
    "README.md",
    "config.toml",
    "__init__.py",
    "config.py",
    "data.py",
    "model.py",
    "training.py",
    "evaluate.py",
    "plotting.py",
    "logging.py",
    "checkpointing.py",
]

DISALLOWED_AUTHOR_NAMES = {
    "student",
    "my name",
    "your full name",
    "your name",
    "author",
}
DISALLOWED_AUTHOR_EMAILS = {
    "noreply@cooper.edu",
    "my@email",
    "your.email@cooper.edu",
    "student@cooper.edu",
    "email@example.com",
}

DEFAULT_OUTPUT_DIR = Path("submissions")


def escape_css_filter(val: str) -> str:
    """Escape a string for safe inclusion in CSS content property."""
    if not val:
        return ""
    return str(val).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def get_jinja_env() -> jinja2.Environment:
    """Initialize Jinja environment with custom filters and templates directory."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["escape_css"] = escape_css_filter
    return env


def _configure_system_library_paths() -> None:
    """Configure dynamic library paths for native C libraries (Pango/Cairo/GLib) on macOS."""
    if sys.platform == "darwin":
        darwin_paths = [
            "/opt/homebrew/lib",
            "/usr/local/lib",
            "/opt/local/lib",
        ]
        current_dyld = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        existing_paths = [p for p in current_dyld.split(":") if p]
        new_paths = [
            p for p in darwin_paths if os.path.exists(p) and p not in existing_paths
        ]
        if new_paths:
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
                new_paths + existing_paths
            )


def build_pdf(html_file: Path, pdf_file: Path) -> bool:
    """Compile an HTML submission file to PDF using WeasyPrint.

    Returns True if successful, False otherwise.
    """
    _configure_system_library_paths()
    try:
        import weasyprint
    except Exception as e:
        print(
            f"\n❌ Error: Failed to compile submission PDF with WeasyPrint: {e}",
            file=sys.stderr,
        )
        print("=" * 72, file=sys.stderr)
        print(
            "PDF compilation requires system-level Pango C libraries.\n"
            "To install them for your operating system:\n\n"
            "  • macOS (Homebrew):    brew install pango\n"
            "  • Ubuntu / Debian:     sudo apt install libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0\n"
            "  • Fedora:              sudo dnf install pango\n"
            "  • Arch Linux:          sudo pacman -S pango\n"
            "  • Windows (WSL2):      sudo apt install libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0\n\n"
            "After installing, re-run:\n"
            "  just submission <hw_name>\n",
            file=sys.stderr,
        )
        return False

    try:
        pdf_file.parent.mkdir(parents=True, exist_ok=True)
        doc = weasyprint.HTML(filename=str(html_file))
        doc.write_pdf(str(pdf_file))
        print(f"==> Successfully generated PDF submission: {pdf_file}")
        return True
    except Exception as e:
        print(f"\n❌ Error generating PDF with WeasyPrint: {e}", file=sys.stderr)
        return False


def get_git_author_info(hw_dir: Path) -> Tuple[str, str]:
    """Retrieve and validate author name and email from git configuration.

    Raises SystemExit if user.name or user.email is missing or placeholder.
    """
    name = ""
    email = ""
    try:
        res_name = subprocess.run(
            ["git", "config", "user.name"],
            cwd=hw_dir if hw_dir.is_dir() else None,
            capture_output=True,
            text=True,
            check=False,
        )
        name = res_name.stdout.strip()
    except Exception:
        pass

    try:
        res_email = subprocess.run(
            ["git", "config", "user.email"],
            cwd=hw_dir if hw_dir.is_dir() else None,
            capture_output=True,
            text=True,
            check=False,
        )
        email = res_email.stdout.strip()
    except Exception:
        pass

    is_name_invalid = not name or name.lower() in DISALLOWED_AUTHOR_NAMES
    is_email_invalid = (
        not email or email.lower() in DISALLOWED_AUTHOR_EMAILS or "@" not in email
    )

    if is_name_invalid or is_email_invalid:
        print("\n" + "=" * 72, file=sys.stderr)
        print(
            "❌ Error: Git author name and email must be configured before building your submission.",
            file=sys.stderr,
        )
        print("=" * 72, file=sys.stderr)
        print(
            "\nPlease configure your full name and Cooper Union email using git:\n",
            file=sys.stderr,
        )
        print('    git config user.name "Your Full Name"', file=sys.stderr)
        print('    git config user.email "your.name@cooper.edu"', file=sys.stderr)
        print(
            "\n(Note: Running these in the repo root configures this local repository without",
            file=sys.stderr,
        )
        print("affecting your global git config.)\n", file=sys.stderr)
        sys.exit(1)

    return name, email


def run_ruff_format(hw_dir: Path) -> None:
    """Format python files in hw_dir using ruff before building submission."""
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    try:
        res = subprocess.run(
            ["uvx", "ruff", "format", str(hw_dir)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            stdout_clean = res.stdout.strip()
            if stdout_clean:
                print(f"==> ruff format: {stdout_clean}")
    except Exception as e:
        print(f"Warning: Could not run ruff format: {e}", file=sys.stderr)


def run_ty_check(hw_dir: Path) -> None:
    """Run ty type checking on hw_dir. Fails build if type errors exist."""
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)

    # Ensure project environment is synced so third-party imports resolve
    try:
        subprocess.run(
            ["uv", "sync", "--project", str(hw_dir), "--quiet"],
            env=env,
            capture_output=True,
            check=False,
        )
    except Exception:
        pass

    cmd = ["uvx", "ty", "check", "--project", str(hw_dir)]
    venv_dir = hw_dir / ".venv"
    if venv_dir.is_dir():
        cmd.extend(["--python", str(venv_dir)])

    try:
        res = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            print(
                "❌ Error: Type checking failed. Fix the following type errors before submitting:",
                file=sys.stderr,
            )
            print("=" * 72, file=sys.stderr)
            if res.stdout:
                print(res.stdout, file=sys.stderr)
            if res.stderr:
                print(res.stderr, file=sys.stderr)
            sys.exit(1)
        else:
            print("==> ty check: All checks passed!")
    except Exception as e:
        print(f"Warning: Could not run ty check: {e}", file=sys.stderr)


def sanitize_id(name: str) -> str:
    """Sanitize string for HTML id attributes."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def get_file_priority(rel_path: Path) -> Tuple[int, int, str]:
    """Calculate sorting priority tuple for a file.

    Returns: (group_priority, module_priority, str(rel_path))
    """
    parts = rel_path.parts
    filename = rel_path.name

    # Group 1: Root metadata files (e.g. pyproject.toml, README.md, config.toml)
    if len(parts) == 1:
        if filename in CORE_MODULE_ORDER:
            return (1, CORE_MODULE_ORDER.index(filename), str(rel_path))
        return (1, 100, str(rel_path))

    # Group 2: Files in src/ package
    if parts[0] == "src":
        if filename in CORE_MODULE_ORDER:
            return (2, CORE_MODULE_ORDER.index(filename), str(rel_path))
        return (2, 100, str(rel_path))

    # Group 3: Other subdirectories
    if filename in CORE_MODULE_ORDER:
        return (3, CORE_MODULE_ORDER.index(filename), str(rel_path))
    return (3, 200, str(rel_path))


def get_lexer(filepath: Path, content: str):
    """Determine Pygments lexer based on filename or content."""
    filename = filepath.name.lower()
    try:
        if filename.endswith(".toml") or filename in ("pyproject.toml", "config.toml"):
            return get_lexer_by_name("toml")
        if filename.endswith(".md"):
            return get_lexer_by_name("markdown")
        if filename.endswith(".py") or filename.endswith(".py.jinja"):
            return get_lexer_by_name("python")
        if filename.endswith(".json"):
            return get_lexer_by_name("json")
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            return get_lexer_by_name("yaml")
        if filename.endswith(".ini") or filename == ".env":
            return get_lexer_by_name("ini")
        return get_lexer_for_filename(filepath.name, content)
    except ClassNotFound:
        return TextLexer()


def clean_svg(svg_content: str) -> str:
    """Clean raw SVG content for direct inlining into HTML.

    Strips XML declaration headers (<?xml ... ?>) and standalone DOCTYPE definitions.
    """
    cleaned = re.sub(r"<\?xml.*?\?>", "", svg_content, flags=re.DOTALL)
    cleaned = re.sub(r"<!DOCTYPE.*?>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def discover_files(hw_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Discover all relevant source files and SVG artifacts within hw_dir."""
    source_files: List[Path] = []
    svg_artifacts: List[Path] = []

    for path in hw_dir.rglob("*"):
        if not path.is_file():
            continue

        # Check ignored directories in parent paths
        rel_to_hw = path.relative_to(hw_dir)
        if any(part in IGNORED_DIRS for part in rel_to_hw.parts):
            continue

        filename = path.name
        if filename.startswith(".#") or filename.endswith("~"):
            continue
        if filename in IGNORED_FILES:
            continue
        if path.suffix.lower() in IGNORED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".svg":
            svg_artifacts.append(rel_to_hw)
        else:
            source_files.append(rel_to_hw)

    # Sort source files by hierarchy priority
    source_files.sort(key=get_file_priority)

    # Sort SVG artifacts (artifacts/ first, then alphabetical)
    def svg_sort_key(p: Path) -> Tuple[int, str]:
        parts = p.parts
        if len(parts) > 1 and parts[0] == "artifacts":
            return (1, str(p))
        return (2, str(p))

    svg_artifacts.sort(key=svg_sort_key)

    return source_files, svg_artifacts


def build_html(hw_dir: Path, output_file: Path) -> None:
    """Build the standalone HTML submission document using Jinja2."""
    if not hw_dir.is_dir():
        print(
            f"Error: Homework directory '{hw_dir}' does not exist or is not a directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    author_name, author_email = get_git_author_info(hw_dir)
    run_ruff_format(hw_dir)
    run_ty_check(hw_dir)

    hw_name = hw_dir.name
    source_file_paths, svg_artifact_paths = discover_files(hw_dir)
    formatter = HtmlFormatter(
        style="friendly", cssclass="highlight", linenos="inline", nowrap=False
    )
    pygments_css = formatter.get_style_defs(".highlight")

    total_lines = 0
    source_files = []

    for rel_path in source_file_paths:
        full_path = hw_dir / rel_path
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"Warning: Could not read {full_path}: {e}", file=sys.stderr)
            continue

        lines = content.splitlines()
        line_count = len(lines)
        total_lines += line_count

        is_markdown = rel_path.suffix.lower() == ".md"
        file_id = f"file-{sanitize_id(str(rel_path))}"

        if is_markdown:
            rendered_content = markdown.markdown(
                content,
                extensions=[
                    "extra",
                    "tables",
                    "fenced_code",
                    "nl2br",
                    "sane_lists",
                    "codehilite",
                ],
                extension_configs={
                    "codehilite": {
                        "css_class": "highlight",
                        "linenums": False,
                        "guess_lang": False,
                    }
                },
            )
            toc_badge = "Markdown"
        else:
            lexer = get_lexer(rel_path, content)
            rendered_content = highlight(content, lexer, formatter)
            toc_badge = f"{line_count} line{'s' if line_count != 1 else ''}"

        source_files.append(
            {
                "id": file_id,
                "path": str(rel_path),
                "line_count": line_count,
                "is_markdown": is_markdown,
                "toc_badge": toc_badge,
                "rendered_content": rendered_content,
            }
        )

    svg_artifacts = []
    for rel_path in svg_artifact_paths:
        full_path = hw_dir / rel_path
        try:
            raw_svg = full_path.read_text(encoding="utf-8", errors="replace")
            cleaned_svg = clean_svg(raw_svg)
        except Exception as e:
            print(f"Warning: Could not read SVG {full_path}: {e}", file=sys.stderr)
            continue

        art_id = f"art-{sanitize_id(str(rel_path))}"
        svg_artifacts.append(
            {
                "id": art_id,
                "path": str(rel_path),
                "cleaned_svg": cleaned_svg,
            }
        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    env = get_jinja_env()
    template = env.get_template("submission.html.jinja")
    html_doc = template.render(
        hw_name=hw_name,
        author_name=author_name,
        author_email=author_email,
        timestamp=timestamp,
        total_lines=total_lines,
        pygments_css=pygments_css,
        source_files=source_files,
        svg_artifacts=svg_artifacts,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_doc, encoding="utf-8")
    print(
        f"==> Successfully generated HTML submission: {output_file} ({total_lines} lines, {len(source_files)} files, {len(svg_artifacts)} plots)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build self-contained HTML/PDF submission for homework assignment."
    )
    parser.add_argument(
        "hw_name", help="Homework assignment directory name (e.g. hw01, example)."
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Custom output file path (defaults to submissions/<hw_name>.html or submissions/<hw_name>.pdf).",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to store generated submissions (defaults to '{DEFAULT_OUTPUT_DIR}').",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Compile submission to PDF via WeasyPrint.",
    )
    args = parser.parse_args()

    hw_dir = Path(args.hw_name).resolve()
    # If the user passed e.g. "example" or "hw01", verify relative to cwd
    if not hw_dir.exists():
        cwd_target = Path.cwd() / args.hw_name
        if cwd_target.exists():
            hw_dir = cwd_target

    if not hw_dir.is_dir():
        print(
            f"❌ Error: Homework directory '{hw_dir}' does not exist or is not a directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        custom_out = Path(args.output).resolve()
        if custom_out.suffix.lower() == ".pdf":
            html_out = custom_out.with_suffix(".html")
            pdf_out = custom_out
            generate_pdf = True
        else:
            html_out = custom_out
            pdf_out = custom_out.with_suffix(".pdf")
            generate_pdf = args.pdf
    else:
        html_out = out_dir / f"{hw_dir.name}.html"
        pdf_out = out_dir / f"{hw_dir.name}.pdf"
        generate_pdf = args.pdf

    build_html(hw_dir, html_out)

    if generate_pdf:
        success = build_pdf(html_out, pdf_out)
        if not success:
            sys.exit(1)
        print("\n" + "=" * 72)
        print(f"📄 Submission Deliverables Ready for '{hw_dir.name}':")
        print(
            f"  • Digital Submission:   Submit '{html_out}' through the course portal."
        )
        print(f"  • Hard Copy Submission: Print '{pdf_out}' for in-person submission.")
        print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
