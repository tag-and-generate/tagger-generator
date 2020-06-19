"""
Compute context based metrics for hypothesis given reference

Usage:
        context_eval.py [options]

Options:
        --hyp=<str>             Path to model hypothesis
        --ref=<str>             Path to model reference
"""

from nlgeval import compute_metrics
from docopt import docopt

class Scorer:

	def __init__(self, ref_file, hyp_file):
		self.ref_file = ref_file
		self.hyp_file = hyp_file
		self.references = list(map(lambda x:x.strip('\n'), open(ref_file, 'r').readlines()))
		self.hypothesis = list(map(lambda x:x.strip('\n'), open(hyp_file, 'r').readlines()))
		self.metrics_dict = {}

	def score(self):
		hyp_test_str = "\n".join([h.replace('\n', '') for h in self.hypothesis])
		ref_test_str = "\n".join([r.replace('\n', '') for r in self.references])
		with open("/tmp/hyp.txt", 'w') as fd_hyp:
			fd_hyp.write(hyp_test_str)
			fd_hyp.close()
		with open("/tmp/ref.txt", 'w') as fd_ref:
			fd_ref.write(ref_test_str)
			fd_ref.close()

		self.metrics_dict = compute_metrics(hypothesis="/tmp/hyp.txt", references=["/tmp/ref.txt"], no_glove=True, no_skipthoughts=True)

	def print_metrics(self):
		for key in self.metrics_dict:
			print (key + "\t\t" + str(self.metrics_dict[key]))

def evaluate():
	args = docopt(__doc__)
	scorer = Scorer(args["--ref"], args["--hyp"])
	scorer.score()
	# scorer.print_metrics()		# Script already prints. Uncomment if needed

if __name__ == '__main__':
	evaluate()
