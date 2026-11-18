"""
Semantics-preserving paraphrases of HDFS log statements, for the controlled format-drift experiments.

Each entry is ONE logging statement in the HDFS source. A statement can produce more than one Drain3
template (E11/E9 and E12/E8 differ only in the number of target datanodes), so rewording a statement
rewords all of its templates together, as a real code change would.

Rules every paraphrase must follow:
  * same event, same meaning, same anomaly label: only the static wording changes;
  * every parameter is kept verbatim (block IDs, IP:port, sizes, file paths, responder numbers, status);
  * the line header (date, time, pid, level, component) is never changed;
  * no new digits are introduced, so Drain3's masking treats parameters exactly as before.

Levels, with magnitude = token edit distance / longer length on masked text (check_paraphrases.py):
  low    : one or two words changed or inserted, same word order             (magnitude <= 0.20)
  medium : several words changed and/or reordered, statement recognisable    (0.40 - 0.60)
  high   : full rewording with a different sentence structure                (>= 0.75)

Eligible statements: those that occur in NORMAL blocks of the test period (format drift rewords
normal events). Statements that occur only in anomalous blocks are left unchanged.
"""
import re

IP = r"\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?"
BLK = r"blk_-?\d+"

STATEMENTS = [
    {"id": "S01", "templates": ["E1"],
     "regex": rf"Receiving block (?P<blk>{BLK}) src: /(?P<src>{IP}) dest: /(?P<dst>{IP})",
     "original": "Receiving block {blk} src: /{src} dest: /{dst}",
     "low": "Receiving block {blk} source: /{src} dest: /{dst}",
     "medium": "Started receiving block {blk} from /{src} to /{dst}",
     "high": "Incoming transfer of {blk} has started with sender /{src} and receiver /{dst}"},
    {"id": "S02", "templates": ["E2"],
     "regex": rf"BLOCK\* NameSystem\.allocateBlock: (?P<path>\S+) (?P<blk>{BLK})",
     "original": "BLOCK* NameSystem.allocateBlock: {path} {blk}",
     "low": "BLOCK* NameSystem.allocateBlock: file {path} {blk}",
     "medium": "BLOCK* NameSystem.allocateBlock: new block for {path} {blk}",
     "high": "New storage block {blk} reserved for file {path}"},
    {"id": "S03", "templates": ["E3"],
     "regex": rf"PacketResponder (?P<n>\d+) for block (?P<blk>{BLK}) (?P<status>.+)",
     "original": "PacketResponder {n} for block {blk} {status}",
     "low": "PacketResponder {n} for block {blk} is {status}",
     "medium": "Responder thread {n} of block {blk} {status}",
     "high": "Acknowledgement handler {n} assigned to {blk} has stopped ({status})"},
    {"id": "S04", "templates": ["E4"],
     "regex": rf"Received block (?P<blk>{BLK}) of size (?P<size>\d+) from /(?P<src>{IP})",
     "original": "Received block {blk} of size {size} from /{src}",
     "low": "Received block {blk} with size {size} from /{src}",
     "medium": "Block {blk} received, size {size}, from /{src}",
     "high": "Finished receiving {size} bytes for {blk} sent by /{src}"},
    {"id": "S05", "templates": ["E5"],
     "regex": rf"BLOCK\* NameSystem\.addStoredBlock: blockMap updated: (?P<node>{IP}) is added to (?P<blk>{BLK}) size (?P<size>\d+)",
     "original": "BLOCK* NameSystem.addStoredBlock: blockMap updated: {node} is added to {blk} size {size}",
     "low": "BLOCK* NameSystem.addStoredBlock: blockMap updated: {node} was added to {blk} size {size}",
     "medium": "BLOCK* NameSystem.addStoredBlock: block map updated: {node} now stores {blk} size {size}",
     "high": "Replica of {blk} with {size} bytes registered on datanode {node}"},
    {"id": "S06", "templates": ["E6"],
     "regex": rf"Received block (?P<blk>{BLK}) src: /(?P<src>{IP}) dest: /(?P<dst>{IP}) of size (?P<size>\d+)",
     "original": "Received block {blk} src: /{src} dest: /{dst} of size {size}",
     "low": "Received block {blk} source: /{src} dest: /{dst} of size {size}",
     "medium": "Finished receiving block {blk} from /{src} to /{dst}, size {size}",
     "high": "Transfer of {blk} completed: {size} bytes moved from /{src} to /{dst}"},
    {"id": "S07", "templates": ["E7"],
     "regex": rf"(?P<node>{IP}):Transmitted block (?P<blk>{BLK}) to /(?P<dst>{IP})",
     "original": "{node}:Transmitted block {blk} to /{dst}",
     "low": "{node}:Sent block {blk} to /{dst}",
     "medium": "{node} has transmitted the block {blk} to /{dst}",
     "high": "Block {blk} was successfully sent from {node} to /{dst}"},
    {"id": "S08", "templates": ["E10"],
     "regex": rf"(?P<node>{IP}) Served block (?P<blk>{BLK}) to /(?P<dst>{IP})",
     "original": "{node} Served block {blk} to /{dst}",
     "low": "{node} Served block {blk} to client /{dst}",
     "medium": "{node} finished serving block {blk} to reader /{dst}",
     "high": "Read request for {blk} completed by {node} for client /{dst}"},
    {"id": "S09", "templates": ["E11", "E9"],
     "regex": rf"BLOCK\* ask (?P<node>{IP}) to replicate (?P<blk>{BLK}) to datanode\(s\) (?P<targets>{IP}(?: {IP})*)",
     "original": "BLOCK* ask {node} to replicate {blk} to datanode(s) {targets}",
     "low": "BLOCK* ask {node} to replicate {blk} to target datanode(s) {targets}",
     "medium": "BLOCK* requesting {node} to make a copy of {blk} on datanode(s) {targets}",
     "high": "Replication scheduled: datanode {node} will send a copy of {blk} onto {targets}"},
    {"id": "S10", "templates": ["E12", "E8"],
     "regex": rf"(?P<node>{IP}) Starting thread to transfer block (?P<blk>{BLK}) to (?P<targets>{IP}(?:, {IP})*)",
     "original": "{node} Starting thread to transfer block {blk} to {targets}",
     "low": "{node} Starting a thread to transfer block {blk} to {targets}",
     "medium": "{node} launching transfer thread for block {blk} to {targets}",
     "high": "Background copy of {blk} from {node} towards {targets} has been started"},
    {"id": "S11", "templates": ["E13"],
     "regex": rf"Verification succeeded for (?P<blk>{BLK})",
     "original": "Verification succeeded for {blk}",
     "low": "Verification succeeded for block {blk}",
     "medium": "Verification of block {blk} succeeded",
     "high": "Integrity scan passed: no corruption found in {blk}"},
    {"id": "S12", "templates": ["E17"],
     "regex": rf"Deleting block (?P<blk>{BLK}) file (?P<path>\S+)",
     "original": "Deleting block {blk} file {path}",
     "low": "Deleting block {blk} at file {path}",
     "medium": "Removing block {blk} stored in {path}",
     "high": "Block file {path} for {blk} is being deleted from disk"},
    {"id": "S13", "templates": ["E32"],
     "regex": rf"BLOCK\* NameSystem\.delete: (?P<blk>{BLK}) is added to invalidSet of (?P<node>{IP})",
     "original": "BLOCK* NameSystem.delete: {blk} is added to invalidSet of {node}",
     "low": "BLOCK* NameSystem.delete: {blk} was added to invalidSet of {node}",
     "medium": "BLOCK* NameSystem.delete: {blk} is marked as invalid on {node}",
     "high": "Scheduled removal of {blk} from datanode {node} (block invalidated)"},
]

LEVELS = ("low", "medium", "high")
for s in STATEMENTS:
    s["_re"] = re.compile(s["regex"] + r"$")


def match(content):
    """Return (statement, params) for a log-message content string, or (None, None)."""
    for s in STATEMENTS:
        m = s["_re"].match(content)
        if m:
            return s, m.groupdict()
    return None, None


def reword(content, level, statement_ids=None):
    """Reword one content string at `level`; return it unchanged if it is not eligible."""
    s, p = match(content)
    if s is None or (statement_ids is not None and s["id"] not in statement_ids):
        return content
    return s[level].format(**p)
