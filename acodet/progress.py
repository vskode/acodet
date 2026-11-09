"""
Progress bars that also work in log files.

tqdm redraws its bar in place with carriage returns, which turns a log file
(e.g. a Slurm job's output) into one long, garbled line. When the output is
not a terminal, `progress` instead writes each update as its own line.
"""
import sys

from tqdm import tqdm

# Seconds between progress lines when the output is not a terminal.
LOG_INTERVAL = 30


class _LineWriter:
    """File for tqdm that writes each refresh as its own line instead of redrawing it."""

    def __init__(self, file):
        self.file = file

    def write(self, s):
        s = s.replace("\r", "").strip()
        if s:
            self.file.write(s + "\n")

    def flush(self):
        self.file.flush()


def progress(*args, file=None, **kwargs):
    """
    tqdm progress bar that redraws in place in a terminal, but prints one line
    per update (every LOG_INTERVAL seconds by default) when the output goes to
    a file, e.g. a Slurm log. Takes the same arguments as tqdm.

    Calls that force a refresh, like set_postfix() with its default
    refresh=True, print a line every time. Pass refresh=False to show the new
    values on the next timed update instead.
    """
    file = file or sys.stderr
    if not getattr(file, "isatty", lambda: False)():
        file = _LineWriter(file)
        kwargs.setdefault("mininterval", LOG_INTERVAL)
        # tqdm forces a refresh after maxinterval (default 10 s), so it must
        # not be shorter than mininterval.
        kwargs.setdefault("maxinterval", kwargs["mininterval"])
    return tqdm(*args, file=file, **kwargs)
