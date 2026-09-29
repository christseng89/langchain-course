from pathlib import Path

import pymupdf

# Both PDFs are located in the same directory as test.py.
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PDF = SCRIPT_DIR / "original.pdf"
OUTPUT_PDF = SCRIPT_DIR / "redacted.pdf"

TARGET_TEXT = "Licensed to Chris Tseng <samfire5200@gmail.com>"
TARGET_PREFIX = "Licensed to Chris Tseng"
TARGET_EMAIL = "samfire5200@gmail.com"


def redact_pdf(input_pdf: Path, output_pdf: Path) -> int:
  if not input_pdf.is_file():
    raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

  if input_pdf.resolve() == output_pdf.resolve():
    raise ValueError("Input and output PDF paths must be different.")

  document = pymupdf.open(input_pdf)

  try:
    if document.needs_pass:
      raise RuntimeError("The PDF is password-protected.")

    if document.page_count == 0:
      raise RuntimeError(
        f"The PDF contains zero pages: {input_pdf}\n"
        "Open the file in a PDF viewer and export or print it "
        "to a new PDF before trying again."
      )

    print(f"Input: {input_pdf}")
    print(f"Pages: {document.page_count}")

    total_redactions = 0

    for page_number, page in enumerate(document, start=1):
      page_redactions = 0

      # First, search for the complete sentence.
      full_matches = page.search_for(
        TARGET_TEXT,
        quads=True,
      )

      for match in full_matches:
        page.add_redact_annot(
          match,
          fill=(1, 1, 1),
          cross_out=False,
        )
        page_redactions += 1

      # Some PDFs store the prefix and email as separate text objects.
      if not full_matches:
        prefix_matches = page.search_for(TARGET_PREFIX)

        email_matches = page.search_for(f"<{TARGET_EMAIL}>")

        if not email_matches:
          email_matches = page.search_for(TARGET_EMAIL)

        for prefix in prefix_matches:
          for email in email_matches:
            prefix_center_y = (prefix.y0 + prefix.y1) / 2

            email_center_y = (email.y0 + email.y1) / 2

            same_line = abs(prefix_center_y - email_center_y) <= max(prefix.height, email.height)

            email_is_after_prefix = email.x0 >= prefix.x0

            if same_line and email_is_after_prefix:
              # Combine both matches into one redaction area.
              area = prefix | email

              # Expand slightly to include angle brackets.
              area.x0 -= 1
              area.x1 += 4
              area.y0 -= 1
              area.y1 += 1

              # Keep the area inside the page.
              area = area & page.rect

              page.add_redact_annot(
                area,
                fill=(1, 1, 1),
                cross_out=False,
              )

              page_redactions += 1
              break

      if page_redactions:
        # Permanently remove content in the marked areas.
        page.apply_redactions()

        total_redactions += page_redactions

        print(f"Page {page_number}: {page_redactions} occurrence(s) removed")

    if total_redactions == 0:
      raise RuntimeError(
        "The target sentence was not found.\n"
        "The text may have different spelling, or the PDF "
        "may contain scanned images instead of selectable text."
      )

    document.save(
      output_pdf,
      garbage=4,
      clean=True,
      deflate=True,
    )

    return total_redactions

  finally:
    document.close()


def main() -> None:
  redaction_count = redact_pdf(
    INPUT_PDF,
    OUTPUT_PDF,
  )

  print()
  print(f"Output: {OUTPUT_PDF}")
  print(f"Total redactions: {redaction_count}")


if __name__ == "__main__":
  main()
