import unittest
import numpy
import random
from sobol_gen import SobolGenerator


class SobolGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.n2 = self.readFileIntoMatrix("test/2d.txt")
        self.n7 = self.readFileIntoMatrix("test/7d.txt")
        self.n16 = self.readFileIntoMatrix("test/16d.txt")

    def compareMatrices(self, first, second):
        self.assertEqual(first.shape, second.shape)

        for i, row in enumerate(first):
            for j in range(len(row)):
                self.assertEqual(
                    round(first[i, j], 6),
                    round(second[i, j], 6),
                    "Difference found at row {0}, column {1}".format(i, j)
                )

    def readFileIntoMatrix(self, filename):
        mat = None

        with open(filename, "r") as inFile:
            for line in inFile:
                if line.startswith("#"):
                    continue
                line = line.strip()
                row = numpy.fromstring(line, sep = " ")

                if mat is None:
                    mat = row
                else:
                    mat = numpy.vstack([mat, row])

        mat = numpy.asmatrix(mat)

        return mat

    def test_2D(self):
        nPoints = 1000
        seed = 0
        n = 2
        s = SobolGenerator(n, seed)

        output = s.generate(nPoints)

        self.compareMatrices(output, self.n2)

    def test_7D(self):
        nPoints = 1000
        seed = 0
        n = 7
        s = SobolGenerator(n, seed)

        output = s.generate(nPoints)

        self.compareMatrices(output, self.n7)

    def test_16d(self):
        nPoints = 1000
        seed = 0
        n = 16
        s = SobolGenerator(n, seed)

        output = s.generate(nPoints)

        self.compareMatrices(output, self.n16)

    def test_skip(self):
        seed = 0
        n = 2

        s = SobolGenerator(n, seed)

        for i in random.sample(range(1000), 1000):
            out = s.element(i)

            s2 = SobolGenerator(n, i)
            out2 = s2.generate(1)

            for j, el in enumerate(out):
                self.assertEqual(round(el, 6), round(self.n2[i, j], 6))
            for j, el in enumerate(out2[0, :]):
                self.assertEqual(round(el, 6), round(self.n2[i, j], 6))

    def test_leap(self):
        leaps = [2, 3, 7]
        n = 2

        for leap in leaps:
            s = SobolGenerator(n, 0, leap)

            nPoints = 1000 // (leap + 1)

            out = s.generate(nPoints)

            for i, vals in enumerate(out):
                index = i * (leap + 1)

                for j, val in enumerate(vals):
                    self.assertEqual(round(val, 6), round(self.n2[index, j], 6))

    def test_skipAndLeap(self):
        leaps = [2, 3, 7]
        skips = [1, 3, 20, 50]
        n = 2

        for leap in leaps:
            for skip in skips:
                nPoints = (1000 - skip) // (leap + 1)

                s = SobolGenerator(n, skip, leap)

                out = s.generate(nPoints)

                for i, vals in enumerate(out):
                    index = i * (leap + 1) + skip

                    for j, val in enumerate(vals):
                        self.assertEqual(round(val, 6), round(self.n2[index, j], 6))


if __name__ == "__main__":
    unittest.main()
