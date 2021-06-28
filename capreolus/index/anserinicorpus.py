from capreolus import constants, get_logger

from . import Index

logger = get_logger(__name__)  # pylint: disable=invalid-name
MAX_THREADS = constants["MAX_THREADS"]

class AnseriniCorpusIndex(Index):
    name = "anserinicorpus"
    indexcorpus = '/GW/NeuralIR/nobackup/anserini0.9-index.clueweb09.englishonly.nostem.stopwording' #TODO hard codeed

    def open(self):
        from jnius import autoclass

        JIndexUtils = autoclass("io.anserini.index.IndexUtils")
        self.index_utils = JIndexUtils(self.indexcorpus)

        JFile = autoclass("java.io.File")
        JFSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
        fsdir = JFSDirectory.open(JFile(self.indexcorpus).toPath())
        self.reader = autoclass("org.apache.lucene.index.DirectoryReader").open(fsdir)
        self.numdocs = self.reader.numDocs()
        self.JTerm = autoclass("org.apache.lucene.index.Term")
        self.numterms = self.reader.getSumTotalTermFreq("contents");

    def get_df(self, term):
        # returns 0 for missing terms
        if not hasattr(self, "reader") or self.reader is None:
            self.open()
        jterm = self.JTerm("contents", term)
        return self.reader.docFreq(jterm)

    def get_tf(self, term):
        # returns 0 for missing terms
        if not hasattr(self, "reader") or self.reader is None:
            self.open()
        jterm = self.JTerm("contents", term)
        return self.reader.totalTermFreq(jterm)
