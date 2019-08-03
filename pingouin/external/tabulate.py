# Copyright (c) 2011-2017 Sergey Astanin [see LICENSE_tabulate.txt]
# -*- coding: utf-8 -*-

"""Pretty-print tabular data."""

from __future__ import print_function
from __future__ import unicode_literals
from collections import namedtuple
from collections.abc import Iterable
from platform import python_version_tuple
import re
import math

if python_version_tuple()[0] < "3":
    from itertools import izip_longest
    from functools import partial
    _none_type = type(None)
    _bool_type = bool
    _int_type = int
    _long_type = long
    _float_type = float
    _text_type = unicode
    _binary_type = str

    def _is_file(f):
        return isinstance(f, file)

else:
    from itertools import zip_longest as izip_longest
    from functools import reduce, partial
    _none_type = type(None)
    _bool_type = bool
    _int_type = int
    _long_type = int
    _float_type = float
    _text_type = str
    _binary_type = bytes
    basestring = str

    import io

    def _is_file(f):
        return isinstance(f, io.IOBase)

try:
    import wcwidth  # optional wide-character (CJK) support
except ImportError:
    wcwidth = None


__all__ = ["tabulate", "tabulate_formats", "simple_separated_format"]
__version__ = "0.8.3"


# minimum extra space in headers
MIN_PADDING = 2

# Whether or not to preserve leading/trailing whitespace in data.
PRESERVE_WHITESPACE = False

_DEFAULT_FLOATFMT = "g"
_DEFAULT_MISSINGVAL = ""


# if True, enable wide-character (CJK) support
WIDE_CHARS_MODE = wcwidth is not None


Line = namedtuple("Line", ["begin", "hline", "sep", "end"])


DataRow = namedtuple("DataRow", ["begin", "sep", "end"])


# A table structure is suppposed to be:
#
#     --- lineabove ---------
#         headerrow
#     --- linebelowheader ---
#         datarow
#     --- linebewteenrows ---
#     ... (more datarows) ...
#     --- linebewteenrows ---
#         last datarow
#     --- linebelow ---------
#
# TableFormat's line* elements can be
#
#   - either None, if the element is not used,
#   - or a Line tuple,
#   - or a function: [col_widths], [col_alignments] -> string.
#
# TableFormat's *row elements can be
#
#   - either None, if the element is not used,
#   - or a DataRow tuple,
#   - or a function: [cell_values], [col_widths], [col_alignments] -> string.
#
# padding (an integer) is the amount of white space around data values.
#
# with_header_hide:
#
#   - either None, to display all table elements unconditionally,
#   - or a list of elements not to be displayed if the table has
#     column headers.
#
TableFormat = namedtuple("TableFormat", ["lineabove", "linebelowheader",
                                         "linebetweenrows", "linebelow",
                                         "headerrow", "datarow",
                                         "padding", "with_header_hide"])


def _pipe_segment_with_colons(align, colwidth):
    """Return a segment of a horizontal line with optional colons which
    indicate column's alignment (as in `pipe` output format)."""
    w = colwidth
    if align in ["right", "decimal"]:
        return ('-' * (w - 1)) + ":"
    elif align == "center":
        return ":" + ('-' * (w - 2)) + ":"
    elif align == "left":
        return ":" + ('-' * (w - 1))
    else:
        return '-' * w


def _pipe_line_with_colons(colwidths, colaligns):
    """Return a horizontal line with optional colons to indicate column's
    alignment (as in `pipe` output format)."""
    segments = [
        _pipe_segment_with_colons(
            a, w) for a, w in zip(
            colaligns, colwidths)]
    return "|" + "|".join(segments) + "|"


def _mediawiki_row_with_attrs(separator, cell_values, colwidths, colaligns):
    alignment = {"left": '',
                 "right": 'align="right"| ',
                 "center": 'align="center"| ',
                 "decimal": 'align="right"| '}
    # hard-coded padding _around_ align attribute and value together
    # rather than padding parameter which affects only the value
    values_with_attrs = [' ' + alignment.get(a, '') + c + ' '
                         for c, a in zip(cell_values, colaligns)]
    colsep = separator * 2
    return (separator + colsep.join(values_with_attrs)).rstrip()


def _textile_row_with_attrs(cell_values, colwidths, colaligns):
    cell_values[0] += ' '
    alignment = {"left": "<.", "right": ">.", "center": "=.", "decimal": ">."}
    values = (alignment.get(a, '') + v for a, v in zip(colaligns, cell_values))
    return '|' + '|'.join(values) + '|'


def _html_begin_table_without_header(colwidths_ignore, colaligns_ignore):
    # this table header will be suppressed if there is a header row
    return "\n".join(["<table>", "<tbody>"])


def _html_row_with_attrs(celltag, cell_values, colwidths, colaligns):
    alignment = {"left": '',
                 "right": ' style="text-align: right;"',
                 "center": ' style="text-align: center;"',
                 "decimal": ' style="text-align: right;"'}
    values_with_attrs = ["<{0}{1}>{2}</{0}>".format(celltag, alignment.get(a, ''), c)
                         for c, a in zip(cell_values, colaligns)]
    rowhtml = "<tr>" + "".join(values_with_attrs).rstrip() + "</tr>"
    if celltag == "th":  # it's a header row, create a new table header
        rowhtml = "\n".join(["<table>",
                             "<thead>",
                             rowhtml,
                             "</thead>",
                             "<tbody>"])
    return rowhtml


def _moin_row_with_attrs(celltag, cell_values, colwidths,
                         colaligns, header=''):
    alignment = {"left": '',
                 "right": '<style="text-align: right;">',
                 "center": '<style="text-align: center;">',
                 "decimal": '<style="text-align: right;">'}
    values_with_attrs = ["{0}{1} {2} ".format(celltag,
                                              alignment.get(a, ''),
                                              header + c + header)
                         for c, a in zip(cell_values, colaligns)]
    return "".join(values_with_attrs) + "||"


def _latex_line_begin_tabular(colwidths, colaligns, booktabs=False):
    alignment = {"left": "l", "right": "r", "center": "c", "decimal": "r"}
    tabular_columns_fmt = "".join([alignment.get(a, "l") for a in colaligns])
    return "\n".join(["\\begin{tabular}{" + tabular_columns_fmt + "}",
                      "\\toprule" if booktabs else "\\hline"])


LATEX_ESCAPE_RULES = {r"&": r"\&", r"%": r"\%", r"$": r"\$", r"#": r"\#",
                      r"_": r"\_", r"^": r"\^{}", r"{": r"\{", r"}": r"\}",
                      r"~": r"\textasciitilde{}", "\\": r"\textbackslash{}",
                      r"<": r"\ensuremath{<}", r">": r"\ensuremath{>}"}


def _latex_row(cell_values, colwidths, colaligns, escrules=LATEX_ESCAPE_RULES):
    def escape_char(c):
        return escrules.get(c, c)
    escaped_values = ["".join(map(escape_char, cell)) for cell in cell_values]
    rowfmt = DataRow("", "&", "\\\\")
    return _build_simple_row(escaped_values, rowfmt)


def _rst_escape_first_column(rows, headers):
    def escape_empty(val):
        if isinstance(val, (_text_type, _binary_type)) and not val.strip():
            return ".."
        else:
            return val
    new_headers = list(headers)
    new_rows = []
    if headers:
        new_headers[0] = escape_empty(headers[0])
    for row in rows:
        new_row = list(row)
        if new_row:
            new_row[0] = escape_empty(row[0])
        new_rows.append(new_row)
    return new_rows, new_headers


_table_formats = {"simple":
                  TableFormat(lineabove=Line("", "-", "  ", ""),
                              linebelowheader=Line("", "-", "  ", ""),
                              linebetweenrows=None,
                              linebelow=Line("", "-", "  ", ""),
                              headerrow=DataRow("", "  ", ""),
                              datarow=DataRow("", "  ", ""),
                              padding=0,
                              with_header_hide=["lineabove", "linebelow"]),
                  "plain":
                  TableFormat(lineabove=None, linebelowheader=None,
                              linebetweenrows=None, linebelow=None,
                              headerrow=DataRow("", "  ", ""),
                              datarow=DataRow("", "  ", ""),
                              padding=0, with_header_hide=None),
                  "grid":
                  TableFormat(lineabove=Line("+", "-", "+", "+"),
                              linebelowheader=Line("+", "=", "+", "+"),
                              linebetweenrows=Line("+", "-", "+", "+"),
                              linebelow=Line("+", "-", "+", "+"),
                              headerrow=DataRow("|", "|", "|"),
                              datarow=DataRow("|", "|", "|"),
                              padding=1, with_header_hide=None),
                  "fancy_grid":
                  TableFormat(lineabove=Line("â•’", "â•", "â•¤", "â••"),
                              linebelowheader=Line("â•ž", "â•", "â•ª", "â•¡"),
                              linebetweenrows=Line("â”œ", "â”€", "â”¼", "â”¤"),
                              linebelow=Line("â•˜", "â•", "â•§", "â•›"),
                              headerrow=DataRow("â”‚", "â”‚", "â”‚"),
                              datarow=DataRow("â”‚", "â”‚", "â”‚"),
                              padding=1, with_header_hide=None),
                  "pipe":
                  TableFormat(lineabove=_pipe_line_with_colons,
                              linebelowheader=_pipe_line_with_colons,
                              linebetweenrows=None,
                              linebelow=None,
                              headerrow=DataRow("|", "|", "|"),
                              datarow=DataRow("|", "|", "|"),
                              padding=1,
                              with_header_hide=["lineabove"]),
                  "orgtbl":
                  TableFormat(lineabove=None,
                              linebelowheader=Line("|", "-", "+", "|"),
                              linebetweenrows=None,
                              linebelow=None,
                              headerrow=DataRow("|", "|", "|"),
                              datarow=DataRow("|", "|", "|"),
                              padding=1, with_header_hide=None),
                  "jira":
                  TableFormat(lineabove=None,
                              linebelowheader=None,
                              linebetweenrows=None,
                              linebelow=None,
                              headerrow=DataRow("||", "||", "||"),
                              datarow=DataRow("|", "|", "|"),
                              padding=1, with_header_hide=None),
                  "presto":
                      TableFormat(lineabove=None,
                                  linebelowheader=Line("", "-", "+", ""),
                                  linebetweenrows=None,
                                  linebelow=None,
                                  headerrow=DataRow("", "|", ""),
                                  datarow=DataRow("", "|", ""),
                                  padding=1, with_header_hide=None),
                  "psql":
                  TableFormat(lineabove=Line("+", "-", "+", "+"),
                              linebelowheader=Line("|", "-", "+", "|"),
                              linebetweenrows=None,
                              linebelow=Line("+", "-", "+", "+"),
                              headerrow=DataRow("|", "|", "|"),
                              datarow=DataRow("|", "|", "|"),
                              padding=1, with_header_hide=None),
                  "rst":
                  TableFormat(lineabove=Line("", "=", "  ", ""),
                              linebelowheader=Line("", "=", "  ", ""),
                              linebetweenrows=None,
                              linebelow=Line("", "=", "  ", ""),
                              headerrow=DataRow("", "  ", ""),
                              datarow=DataRow("", "  ", ""),
                              padding=0, with_header_hide=None),
                  "mediawiki":
                  TableFormat(lineabove=Line("{| class=\"wikitable\" style=\"text-align: left;\"",
                                             "", "", "\n|+ <!-- caption -->\n|-"),
                              linebelowheader=Line("|-", "", "", ""),
                              linebetweenrows=Line("|-", "", "", ""),
                              linebelow=Line("|}", "", "", ""),
                              headerrow=partial(
                      _mediawiki_row_with_attrs, "!"),
                      datarow=partial(_mediawiki_row_with_attrs, "|"),
                      padding=0, with_header_hide=None),
                  "moinmoin":
                  TableFormat(lineabove=None,
                              linebelowheader=None,
                              linebetweenrows=None,
                              linebelow=None,
                              headerrow=partial(
                                  _moin_row_with_attrs, "||", header="'''"),
                              datarow=partial(_moin_row_with_attrs, "||"),
                              padding=1, with_header_hide=None),
                  "youtrack":
                  TableFormat(lineabove=None,
                              linebelowheader=None,
                              linebetweenrows=None,
                              linebelow=None,
                              headerrow=DataRow("|| ", " || ", " || "),
                              datarow=DataRow("| ", " | ", " |"),
                              padding=1, with_header_hide=None),
                  "html":
                  TableFormat(lineabove=_html_begin_table_without_header,
                              linebelowheader="",
                              linebetweenrows=None,
                              linebelow=Line("</tbody>\n</table>", "", "", ""),
                              headerrow=partial(_html_row_with_attrs, "th"),
                              datarow=partial(_html_row_with_attrs, "td"),
                              padding=0, with_header_hide=["lineabove"]),
                  "latex":
                  TableFormat(lineabove=_latex_line_begin_tabular,
                              linebelowheader=Line("\\hline", "", "", ""),
                              linebetweenrows=None,
                              linebelow=Line(
                                  "\\hline\n\\end{tabular}", "", "", ""),
                              headerrow=_latex_row,
                              datarow=_latex_row,
                              padding=1, with_header_hide=None),
                  "latex_raw":
                  TableFormat(lineabove=_latex_line_begin_tabular,
                              linebelowheader=Line("\\hline", "", "", ""),
                              linebetweenrows=None,
                              linebelow=Line(
                                  "\\hline\n\\end{tabular}", "", "", ""),
                              headerrow=partial(_latex_row, escrules={}),
                              datarow=partial(_latex_row, escrules={}),
                              padding=1, with_header_hide=None),
                  "latex_booktabs":
                  TableFormat(lineabove=partial(_latex_line_begin_tabular, booktabs=True),
                              linebelowheader=Line("\\midrule", "", "", ""),
                              linebetweenrows=None,
                              linebelow=Line(
                      "\\bottomrule\n\\end{tabular}", "", "", ""),
                      headerrow=_latex_row,
                      datarow=_latex_row,
                      padding=1, with_header_hide=None),
                  "tsv":
                  TableFormat(lineabove=None, linebelowheader=None,
                              linebetweenrows=None, linebelow=None,
                              headerrow=DataRow("", "\t", ""),
                              datarow=DataRow("", "\t", ""),
                              padding=0, with_header_hide=None),
                  "textile":
                  TableFormat(lineabove=None, linebelowheader=None,
                              linebetweenrows=None, linebelow=None,
                              headerrow=DataRow("|_. ", "|_.", "|"),
                              datarow=_textile_row_with_attrs,
                              padding=1, with_header_hide=None)}


tabulate_formats = list(sorted(_table_formats.keys()))

# The table formats for which multiline cells will be folded into subsequent
# table rows. The key is the original format specified at the API. The value is
# the format that will be used to represent the original format.
multiline_formats = {
    "plain": "plain",
    "simple": "simple",
    "grid": "grid",
    "fancy_grid": "fancy_grid",
    "pipe": "pipe",
    "orgtbl": "orgtbl",
    "jira": "jira",
    "presto": "presto",
    "psql": "psql",
    "rst": "rst",
}

_multiline_codes = re.compile(r"\r|\n|\r\n")
_multiline_codes_bytes = re.compile(b"\r|\n|\r\n")
# ANSI color codes
_invisible_codes = re.compile(r"\x1b\[\d+[;\d]*m|\x1b\[\d*\;\d*\;\d*m")
# ANSI color codes
_invisible_codes_bytes = re.compile(b"\x1b\[\d+[;\d]*m|\x1b\[\d*\;\d*\;\d*m")


def simple_separated_format(separator):
    """Construct a simple TableFormat with columns separated by a separator.
    """
    return TableFormat(None, None, None, None,
                       headerrow=DataRow('', separator, ''),
                       datarow=DataRow('', separator, ''),
                       padding=0, with_header_hide=None)


def _isconvertible(conv, string):
    try:
        n = conv(string)
        return True
    except (ValueError, TypeError):
        return False


def _isnumber(string):
    """
    >>> _isnumber("123.45")
    True
    >>> _isnumber("123")
    True
    >>> _isnumber("spam")
    False
    >>> _isnumber("123e45678")
    False
    >>> _isnumber("inf")
    True
    """
    if not _isconvertible(float, string):
        return False
    elif isinstance(string, (_text_type, _binary_type)) and (
            math.isinf(float(string)) or math.isnan(float(string))):
        return string.lower() in ['inf', '-inf', 'nan']
    return True


def _isint(string, inttype=int):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return isinstance(string, inttype) or\
        (isinstance(string, _binary_type) or isinstance(string, _text_type))\
        and\
        _isconvertible(inttype, string)


def _isbool(string):
    """
    >>> _isbool(True)
    True
    >>> _isbool("False")
    True
    >>> _isbool(1)
    False
    """
    return isinstance(string, _bool_type) or\
        (isinstance(string, (_binary_type, _text_type))
         and
         string in ("True", "False"))


def _type(string, has_invisible=True, numparse=True):
    """The least generic type (type(None), int, float, str, unicode).

    >>> _type(None) is type(None)
    True
    >>> _type("foo") is type("")
    True
    >>> _type("1") is type(1)
    True
    >>> _type('\x1b[31m42\x1b[0m') is type(42)
    True
    >>> _type('\x1b[31m42\x1b[0m') is type(42)
    True

    """

    if has_invisible and \
       (isinstance(string, _text_type) or isinstance(string, _binary_type)):
        string = _strip_invisible(string)

    if string is None:
        return _none_type
    elif hasattr(string, "isoformat"):  # datetime.datetime, date, and time
        return _text_type
    elif _isbool(string):
        return _bool_type
    elif _isint(string) and numparse:
        return int
    elif _isint(string, _long_type) and numparse:
        return int
    elif _isnumber(string) and numparse:
        return float
    elif isinstance(string, _binary_type):
        return _binary_type
    else:
        return _text_type


def _afterpoint(string):
    """Symbols after a decimal point, -1 if the string lacks the decimal point.

    >>> _afterpoint("123.45")
    2
    >>> _afterpoint("1001")
    -1
    >>> _afterpoint("eggs")
    -1
    >>> _afterpoint("123e45")
    2

    """
    if _isnumber(string):
        if _isint(string):
            return -1
        else:
            pos = string.rfind(".")
            pos = string.lower().rfind("e") if pos < 0 else pos
            if pos >= 0:
                return len(string) - pos - 1
            else:
                return -1  # no point
    else:
        return -1  # not a number


def _padleft(width, s):
    """Flush right.
    """
    fmt = "{0:>%ds}" % width
    return fmt.format(s)


def _padright(width, s):
    """Flush left.
    """
    fmt = "{0:<%ds}" % width
    return fmt.format(s)


def _padboth(width, s):
    """Center string.
    """
    fmt = "{0:^%ds}" % width
    return fmt.format(s)


def _padnone(ignore_width, s):
    return s


def _strip_invisible(s):
    "Remove invisible ANSI color codes."
    if isinstance(s, _text_type):
        return re.sub(_invisible_codes, "", s)
    else:  # a bytestring
        return re.sub(_invisible_codes_bytes, "", s)


def _visible_width(s):
    """Visible width of a printed string. ANSI color codes are removed.

    >>> _visible_width('\x1b[31mhello\x1b[0m'), _visible_width("world")
    (5, 5)

    """
    # optional wide-character support
    if wcwidth is not None and WIDE_CHARS_MODE:
        len_fn = wcwidth.wcswidth
    else:
        len_fn = len
    if isinstance(s, _text_type) or isinstance(s, _binary_type):
        return len_fn(_strip_invisible(s))
    else:
        return len_fn(_text_type(s))


def _is_multiline(s):
    if isinstance(s, _text_type):
        return bool(re.search(_multiline_codes, s))
    else:  # a bytestring
        return bool(re.search(_multiline_codes_bytes, s))


def _multiline_width(multiline_s, line_width_fn=len):
    """Visible width of a potentially multiline content."""
    return max(map(line_width_fn, re.split("[\r\n]", multiline_s)))


def _choose_width_fn(has_invisible, enable_widechars, is_multiline):
    """Return a function to calculate visible cell width."""
    if has_invisible:
        line_width_fn = _visible_width
    elif enable_widechars:  # optional wide-character support if available
        line_width_fn = wcwidth.wcswidth
    else:
        line_width_fn = len
    if is_multiline:
        def width_fn(s): return _multiline_width(s, line_width_fn)
    else:
        width_fn = line_width_fn
    return width_fn


def _align_column_choose_padfn(strings, alignment, has_invisible):
    if alignment == "right":
        if not PRESERVE_WHITESPACE:
            strings = [s.strip() for s in strings]
        padfn = _padleft
    elif alignment == "center":
        if not PRESERVE_WHITESPACE:
            strings = [s.strip() for s in strings]
        padfn = _padboth
    elif alignment == "decimal":
        if has_invisible:
            decimals = [_afterpoint(_strip_invisible(s)) for s in strings]
        else:
            decimals = [_afterpoint(s) for s in strings]
        maxdecimals = max(decimals)
        strings = [s + (maxdecimals - decs) * " "
                   for s, decs in zip(strings, decimals)]
        padfn = _padleft
    elif not alignment:
        padfn = _padnone
    else:
        if not PRESERVE_WHITESPACE:
            strings = [s.strip() for s in strings]
        padfn = _padright
    return strings, padfn


def _align_column(strings, alignment, minwidth=0,
                  has_invisible=True, enable_widechars=False, is_multiline=False):
    """[string] -> [padded_string]"""
    strings, padfn = _align_column_choose_padfn(
        strings, alignment, has_invisible)
    width_fn = _choose_width_fn(has_invisible, enable_widechars, is_multiline)

    s_widths = list(map(width_fn, strings))
    maxwidth = max(max(s_widths), minwidth)
    # TODO: refactor column alignment in single-line and multiline modes
    if is_multiline:
        if not enable_widechars and not has_invisible:
            padded_strings = [
                "\n".join([padfn(maxwidth, s) for s in ms.splitlines()])
                for ms in strings]
        else:
            # enable wide-character width corrections
            s_lens = [max((len(s) for s in re.split("[\r\n]", ms)))
                      for ms in strings]
            visible_widths = [maxwidth - (w - l)
                              for w, l in zip(s_widths, s_lens)]
            # wcswidth and _visible_width don't count invisible characters;
            # padfn doesn't need to apply another correction
            padded_strings = ["\n".join([padfn(w, s) for s in (ms.splitlines() or ms)])
                              for ms, w in zip(strings, visible_widths)]
    else:  # single-line cell values
        if not enable_widechars and not has_invisible:
            padded_strings = [padfn(maxwidth, s) for s in strings]
        else:
            # enable wide-character width corrections
            s_lens = list(map(len, strings))
            visible_widths = [maxwidth - (w - l)
                              for w, l in zip(s_widths, s_lens)]
            # wcswidth and _visible_width don't count invisible characters;
            # padfn doesn't need to apply another correction
            padded_strings = [
                padfn(
                    w, s) for s, w in zip(
                    strings, visible_widths)]
    return padded_strings


def _more_generic(type1, type2):
    types = {_none_type: 0, _bool_type: 1, int: 2, float: 3,
             _binary_type: 4, _text_type: 5}
    invtypes = {5: _text_type, 4: _binary_type, 3: float, 2: int,
                1: _bool_type, 0: _none_type}
    moregeneric = max(types.get(type1, 5), types.get(type2, 5))
    return invtypes[moregeneric]


def _column_type(strings, has_invisible=True, numparse=True):
    """The least generic type all column values are convertible to.

    >>> _column_type([True, False]) is _bool_type
    True
    >>> _column_type(["1", "2"]) is _int_type
    True
    >>> _column_type(["1", "2.3"]) is _float_type
    True
    >>> _column_type(["1", "2.3", "four"]) is _text_type
    True
    >>> _column_type(["four", '\u043f\u044f\u0442\u044c']) is _text_type
    True
    >>> _column_type([None, "brux"]) is _text_type
    True
    >>> _column_type([1, 2, None]) is _int_type
    True
    >>> import datetime as dt
    >>> _column_type([dt.datetime(1991,2,19), dt.time(17,35)]) is _text_type
    True

    """
    types = [_type(s, has_invisible, numparse) for s in strings]
    return reduce(_more_generic, types, _bool_type)


def _format(val, valtype, floatfmt, missingval="", has_invisible=True):
    """Format a value accoding to its type.
    """
    if val is None:
        return missingval

    if valtype in [int, _text_type]:
        return "{0}".format(val)
    elif valtype is _binary_type:
        try:
            return _text_type(val, "ascii")
        except TypeError:
            return _text_type(val)
    elif valtype is float:
        is_a_colored_number = has_invisible and isinstance(val,
                                                           (_text_type, _binary_type))
        if is_a_colored_number:
            raw_val = _strip_invisible(val)
            formatted_val = format(float(raw_val), floatfmt)
            return val.replace(raw_val, formatted_val)
        else:
            return format(float(val), floatfmt)
    else:
        return "{0}".format(val)


def _align_header(header, alignment, width, visible_width, is_multiline=False,
                  width_fn=None):
    "Pad string header to width chars given known visible_width of the header."
    if is_multiline:
        header_lines = re.split(_multiline_codes, header)
        padded_lines = [_align_header(h, alignment, width, width_fn(h))
                        for h in header_lines]
        return "\n".join(padded_lines)
    # else: not multiline
    ninvisible = len(header) - visible_width
    width += ninvisible
    if alignment == "left":
        return _padright(width, header)
    elif alignment == "center":
        return _padboth(width, header)
    elif not alignment:
        return "{0}".format(header)
    else:
        return _padleft(width, header)


def _prepend_row_index(rows, index):
    """Add a left-most index column."""
    if index is None or index is False:
        return rows
    if len(index) != len(rows):
        print('index=', index)
        print('rows=', rows)
        raise ValueError('index must be as long as the number of data rows')
    rows = [[v] + list(row) for v, row in zip(index, rows)]
    return rows


def _bool(val):
    "A wrapper around standard bool() which doesn't throw on NumPy arrays"
    try:
        return bool(val)
    except ValueError:  # val is likely to be a numpy array with many elements
        return False


def _normalize_tabular_data(tabular_data, headers, showindex="default"):
    """Transform a supported data type to a list of lists,
    and a list of headers.

    Supported tabular data types:

    * list-of-lists or another iterable of iterables

    * list of named tuples (usually used with headers="keys")

    * list of dicts (usually used with headers="keys")

    * list of OrderedDicts (usually used with headers="keys")

    * 2D NumPy arrays

    * NumPy record arrays (usually used with headers="keys")

    * dict of iterables (usually used with headers="keys")

    * pandas.DataFrame (usually used with headers="keys")

    The first row can be used as headers if headers="firstrow",
    column indices can be used as headers if headers="keys".

    If showindex="default", show row indices of the pandas.DataFrame.
    If showindex="always", show row indices for all types of data.
    If showindex="never", don't show row indices for all types of data.
    If showindex is an iterable, show its values as row indices.

    """

    try:
        bool(headers)
        is_headers2bool_broken = False
    except ValueError:  # numpy.ndarray, pandas.core.index.Index, ...
        is_headers2bool_broken = True
        headers = list(headers)

    index = None
    if hasattr(tabular_data, "keys") and hasattr(tabular_data, "values"):
        # dict-like and pandas.DataFrame?
        if hasattr(tabular_data.values, "__call__"):
            # likely a conventional dict
            keys = tabular_data.keys()
            # columns have to be transposed
            rows = list(izip_longest(*tabular_data.values()))
        elif hasattr(tabular_data, "index"):
            # values is a property, has .index => it's likely a
            # pandas.DataFrame (pandas 0.11.0)
            keys = list(tabular_data)
            if tabular_data.index.name is not None:
                if isinstance(tabular_data.index.name, list):
                    keys[:0] = tabular_data.index.name
                else:
                    keys[:0] = [tabular_data.index.name]
            vals = tabular_data.values  # values matrix doesn't need to be transposed
            # for DataFrames add an index per default
            index = list(tabular_data.index)
            rows = [list(row) for row in vals]
        else:
            raise ValueError(
                "tabular data doesn't appear to be a dict or a DataFrame")

        if headers == "keys":
            headers = list(map(_text_type, keys))  # headers should be strings

    else:  # it's a usual an iterable of iterables, or a NumPy array
        rows = list(tabular_data)

        if (headers == "keys" and not rows):
            # an empty table (issue #81)
            headers = []
        elif (headers == "keys" and
              hasattr(tabular_data, "dtype") and
              getattr(tabular_data.dtype, "names")):
            # numpy record array
            headers = tabular_data.dtype.names
        elif (headers == "keys"
              and len(rows) > 0
              and isinstance(rows[0], tuple)
              and hasattr(rows[0], "_fields")):
            # namedtuple
            headers = list(map(_text_type, rows[0]._fields))
        elif (len(rows) > 0
              and isinstance(rows[0], dict)):
            # dict or OrderedDict
            uniq_keys = set()  # implements hashed lookup
            keys = []  # storage for set
            if headers == "firstrow":
                firstdict = rows[0] if len(rows) > 0 else {}
                keys.extend(firstdict.keys())
                uniq_keys.update(keys)
                rows = rows[1:]
            for row in rows:
                for k in row.keys():
                    # Save unique items in input order
                    if k not in uniq_keys:
                        keys.append(k)
                        uniq_keys.add(k)
            if headers == 'keys':
                headers = keys
            elif isinstance(headers, dict):
                # a dict of headers for a list of dicts
                headers = [headers.get(k, k) for k in keys]
                headers = list(map(_text_type, headers))
            elif headers == "firstrow":
                if len(rows) > 0:
                    headers = [firstdict.get(k, k) for k in keys]
                    headers = list(map(_text_type, headers))
                else:
                    headers = []
            elif headers:
                raise ValueError(
                    'headers for a list of dicts is not a dict or a keyword')
            rows = [[row.get(k) for k in keys] for row in rows]

        elif (headers == "keys"
              and hasattr(tabular_data, "description")
              and hasattr(tabular_data, "fetchone")
              and hasattr(tabular_data, "rowcount")):
            # Python Database API cursor object (PEP 0249)
            # print tabulate(cursor, headers='keys')
            headers = [column[0] for column in tabular_data.description]

        elif headers == "keys" and len(rows) > 0:
            # keys are column indices
            headers = list(map(_text_type, range(len(rows[0]))))

    # take headers from the first row if necessary
    if headers == "firstrow" and len(rows) > 0:
        if index is not None:
            headers = [index[0]] + list(rows[0])
            index = index[1:]
        else:
            headers = rows[0]
        headers = list(map(_text_type, headers))  # headers should be strings
        rows = rows[1:]

    headers = list(map(_text_type, headers))
    rows = list(map(list, rows))

    # add or remove an index column
    showindex_is_a_str = type(showindex) in [_text_type, _binary_type]
    if showindex == "default" and index is not None:
        rows = _prepend_row_index(rows, index)
    elif isinstance(showindex, Iterable) and not showindex_is_a_str:
        rows = _prepend_row_index(rows, list(showindex))
    elif showindex == "always" or (_bool(showindex) and not showindex_is_a_str):
        if index is None:
            index = list(range(len(rows)))
        rows = _prepend_row_index(rows, index)
    elif showindex == "never" or (not _bool(showindex) and not showindex_is_a_str):
        pass

    # pad with empty headers for initial columns if necessary
    if headers and len(rows) > 0:
        nhs = len(headers)
        ncols = len(rows[0])
        if nhs < ncols:
            headers = [""] * (ncols - nhs) + headers

    return rows, headers


def tabulate(tabular_data, headers=(), tablefmt="simple",
             floatfmt=_DEFAULT_FLOATFMT, numalign="decimal", stralign="left",
             missingval=_DEFAULT_MISSINGVAL, showindex="default",
             disable_numparse=False):
    """Format a fixed width table for pretty printing.
    """
    if tabular_data is None:
        tabular_data = []
    list_of_lists, headers = _normalize_tabular_data(
        tabular_data, headers, showindex=showindex)

    # empty values in the first column of RST tables should be escaped (issue #82)
    # "" should be escaped as "\\ " or ".."
    if tablefmt == 'rst':
        list_of_lists, headers = _rst_escape_first_column(
            list_of_lists, headers)

    # optimization: look for ANSI control codes once,
    # enable smart width functions only if a control code is found
    plain_text = '\t'.join(['\t'.join(map(_text_type, headers))] +
                           ['\t'.join(map(_text_type, row)) for row in list_of_lists])

    has_invisible = re.search(_invisible_codes, plain_text)
    enable_widechars = wcwidth is not None and WIDE_CHARS_MODE
    if tablefmt in multiline_formats and _is_multiline(plain_text):
        tablefmt = multiline_formats.get(tablefmt, tablefmt)
        is_multiline = True
    else:
        is_multiline = False
    width_fn = _choose_width_fn(has_invisible, enable_widechars, is_multiline)

    # format rows and columns, convert numeric values to strings
    cols = list(izip_longest(*list_of_lists))
    numparses = _expand_numparse(disable_numparse, len(cols))
    coltypes = [_column_type(col, numparse=np) for col, np in
                zip(cols, numparses)]
    if isinstance(floatfmt, basestring):  # old version
        # just duplicate the string to use in each column
        float_formats = len(cols) * [floatfmt]
    else:  # if floatfmt is list, tuple etc we have one per column
        float_formats = list(floatfmt)
        if len(float_formats) < len(cols):
            float_formats.extend(
                (len(cols) - len(float_formats)) * [_DEFAULT_FLOATFMT])
    if isinstance(missingval, basestring):
        missing_vals = len(cols) * [missingval]
    else:
        missing_vals = list(missingval)
        if len(missing_vals) < len(cols):
            missing_vals.extend(
                (len(cols) - len(missing_vals)) * [_DEFAULT_MISSINGVAL])
    cols = [[_format(v, ct, fl_fmt, miss_v, has_invisible) for v in c]
            for c, ct, fl_fmt, miss_v in zip(cols, coltypes, float_formats, missing_vals)]

    # align columns
    aligns = [numalign if ct in [int, float] else stralign for ct in coltypes]
    minwidths = [
        width_fn(h) + MIN_PADDING for h in headers] if headers else [0] * len(cols)
    cols = [_align_column(c, a, minw, has_invisible, enable_widechars, is_multiline)
            for c, a, minw in zip(cols, aligns, minwidths)]

    if headers:
        # align headers and add headers
        t_cols = cols or [['']] * len(headers)
        t_aligns = aligns or [stralign] * len(headers)
        minwidths = [max(minw, max(width_fn(cl) for cl in c))
                     for minw, c in zip(minwidths, t_cols)]
        headers = [_align_header(h, a, minw, width_fn(h), is_multiline, width_fn)
                   for h, a, minw in zip(headers, t_aligns, minwidths)]
        rows = list(zip(*cols))
    else:
        minwidths = [max(width_fn(cl) for cl in c) for c in cols]
        rows = list(zip(*cols))

    if not isinstance(tablefmt, TableFormat):
        tablefmt = _table_formats.get(tablefmt, _table_formats["simple"])

    return _format_table(tablefmt, headers, rows,
                         minwidths, aligns, is_multiline)


def _expand_numparse(disable_numparse, column_count):
    """
    Return a list of bools of length `column_count` which indicates whether
    number parsing should be used on each column.
    If `disable_numparse` is a list of indices, each of those indices are False,
    and everything else is True.
    If `disable_numparse` is a bool, then the returned list is all the same.
    """
    if isinstance(disable_numparse, Iterable):
        numparses = [True] * column_count
        for index in disable_numparse:
            numparses[index] = False
        return numparses
    else:
        return [not disable_numparse] * column_count


def _pad_row(cells, padding):
    if cells:
        pad = " " * padding
        padded_cells = [pad + cell + pad for cell in cells]
        return padded_cells
    else:
        return cells


def _build_simple_row(padded_cells, rowfmt):
    "Format row according to DataRow format without padding."
    begin, sep, end = rowfmt
    return (begin + sep.join(padded_cells) + end).rstrip()


def _build_row(padded_cells, colwidths, colaligns, rowfmt):
    "Return a string which represents a row of data cells."
    if not rowfmt:
        return None
    if hasattr(rowfmt, "__call__"):
        return rowfmt(padded_cells, colwidths, colaligns)
    else:
        return _build_simple_row(padded_cells, rowfmt)


def _append_basic_row(lines, padded_cells, colwidths, colaligns, rowfmt):
    lines.append(_build_row(padded_cells, colwidths, colaligns, rowfmt))
    return lines


def _append_multiline_row(lines, padded_multiline_cells,
                          padded_widths, colaligns, rowfmt, pad):
    colwidths = [w - 2 * pad for w in padded_widths]
    cells_lines = [c.splitlines() for c in padded_multiline_cells]
    nlines = max(map(len, cells_lines))  # number of lines in the row
    # vertically pad cells where some lines are missing
    cells_lines = [(cl + [' ' * w] * (nlines - len(cl)))
                   for cl, w in zip(cells_lines, colwidths)]
    lines_cells = [[cl[i] for cl in cells_lines] for i in range(nlines)]
    for ln in lines_cells:
        padded_ln = _pad_row(ln, pad)
        _append_basic_row(lines, padded_ln, colwidths, colaligns, rowfmt)
    return lines


def _build_line(colwidths, colaligns, linefmt):
    "Return a string which represents a horizontal line."
    if not linefmt:
        return None
    if hasattr(linefmt, "__call__"):
        return linefmt(colwidths, colaligns)
    else:
        begin, fill, sep, end = linefmt
        cells = [fill * w for w in colwidths]
        return _build_simple_row(cells, (begin, sep, end))


def _append_line(lines, colwidths, colaligns, linefmt):
    lines.append(_build_line(colwidths, colaligns, linefmt))
    return lines


def _format_table(fmt, headers, rows, colwidths, colaligns, is_multiline):
    """Produce a plain-text representation of the table."""
    lines = []
    hidden = fmt.with_header_hide if (headers and fmt.with_header_hide) else []
    pad = fmt.padding
    headerrow = fmt.headerrow

    padded_widths = [(w + 2 * pad) for w in colwidths]
    if is_multiline:
        # do it later, in _append_multiline_row
        def pad_row(row, _): return row
        append_row = partial(_append_multiline_row, pad=pad)
    else:
        pad_row = _pad_row
        append_row = _append_basic_row

    padded_headers = pad_row(headers, pad)
    padded_rows = [pad_row(row, pad) for row in rows]

    if fmt.lineabove and "lineabove" not in hidden:
        _append_line(lines, padded_widths, colaligns, fmt.lineabove)

    if padded_headers:
        append_row(lines, padded_headers, padded_widths, colaligns, headerrow)
        if fmt.linebelowheader and "linebelowheader" not in hidden:
            _append_line(lines, padded_widths, colaligns, fmt.linebelowheader)

    if padded_rows and fmt.linebetweenrows and "linebetweenrows" not in hidden:
        # initial rows with a line below
        for row in padded_rows[:-1]:
            append_row(lines, row, padded_widths, colaligns, fmt.datarow)
            _append_line(lines, padded_widths, colaligns, fmt.linebetweenrows)
        # the last row without a line below
        append_row(lines,
                   padded_rows[-1],
                   padded_widths,
                   colaligns,
                   fmt.datarow)
    else:
        for row in padded_rows:
            append_row(lines, row, padded_widths, colaligns, fmt.datarow)

    if fmt.linebelow and "linebelow" not in hidden:
        _append_line(lines, padded_widths, colaligns, fmt.linebelow)

    if headers or rows:
        return "\n".join(lines)
    else:  # a completely empty table
        return ""
