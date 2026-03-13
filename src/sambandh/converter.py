"""Docling converter factory -- loaded once per worker process."""

import os

os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("ORT_NUM_INTER_THREADS", "1")
os.environ.setdefault("ORT_NUM_INTRA_THREADS", "1")

_converter = None


def _make_converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    ocr_options = TesseractCliOcrOptions(
        force_full_page_ocr=True,
        lang=["mal", "eng"],
    )
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = ocr_options
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def warm_converter():
    """Initialise the global converter. Called once per worker child process."""
    global _converter
    if _converter is None:
        _converter = _make_converter()
    return _converter


def get_converter():
    """Return the pre-warmed converter, creating it if needed."""
    global _converter
    if _converter is None:
        _converter = _make_converter()
    return _converter
