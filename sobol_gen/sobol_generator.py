import numpy
from .utility import highBitPos, lowBitPos


class SobolGenerator(object):
    """Handles all aspects of Sobol' sequence generation."""

    nMax = 40
    nMin = 1
    logMax = 30
    intTypeString = "i4"
    intType = numpy.int32

    def __init__(
        self,
        n,
        seed = None,
        leap = 0
    ):
        self.n = n

        if seed is None:
            self.seed = SobolGenerator.goodSeeds[self.n]
        else:
            self.seed = SobolGenerator.intType(seed)

        self.leap = leap

        self.lastQ = numpy.zeros(self.n, SobolGenerator.intTypeString)

        for i in range(self.seed):
            self.lastQ = self.nextQ(self.lastQ, i)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        if value < SobolGenerator.nMin or value > SobolGenerator.nMax:
            raise RuntimeError(
                "Number of dimensions specified outside of range [{0}, {1}]".format(
                    SobolGenerator.nMin, SobolGenerator.nMax
                )
            )

        self._n = value
        self.v = numpy.copy(SobolGenerator.baseV)

        for i in range(1, self.n):
            j = SobolGenerator.poly[i]
            m = 0
            j //= 2
            while j > 0:
                j //= 2
                m += 1
            j = SobolGenerator.poly[i]
            include = numpy.full(m, False, bool)

            for k in reversed(range(m)):
                j2 = j // 2
                include[k] = (j != 2 * j2)
                j = j2

            for j in range(m, SobolGenerator.maxCol):
                newV = self.v[i, j - m]
                l = 1
                for k in range(m):
                    l *= 2
                    if include[k]:
                        newV = numpy.bitwise_xor(
                            newV, l * self.v[i, j - k - 1]
                        )
                self.v[i, j] = newV

        l = 1
        for j in reversed(range(0, SobolGenerator.maxCol - 1)):
            l *= 2
            self.v[:, j] *= l

        self.recipD = 1.0 / float(2 * l)

    def generate(self, N, seed = None, leap = None):
        """Generates a Sobol' sequence.

        Keyword arguments:
        N -- How many elements to generate
        seed -- The seed to start from (defaults to last seed used, or 0)
        leap -- The number of elements to ignore for every element taken (defaults
            to value set in object, or 0)
        """

        if seed is None:
            seed = self.seed
        if leap is None:
            leap = self.leap

        matrix = numpy.full((N, self.n), numpy.nan)
        nextOutput = seed
        numberOutput = 0

        while numberOutput < N:
            element = self.element(seed)

            if seed == nextOutput:
                matrix[numberOutput, :] = element
                nextOutput += leap + 1
                numberOutput += 1

            seed += 1

        return matrix

    def element(self, seed):
        """Generates a single element of a Sobol' sequence.

        Keyword arguments:
        seed - The seed to use to generate the element
        """

        if seed == 0:
            self.seed = 0
            self.lastQ = numpy.zeros(self.n, SobolGenerator.intTypeString)
        elif seed > self.seed:
            for i in range(self.seed, seed):
                self.lastQ = self.nextQ(self.lastQ, self.seed)
                self.seed += 1
        elif seed < self.seed:
            self.lastQ = numpy.zeros(self.n, SobolGenerator.intTypeString)
            self.seed = 0
            for i in range(0, seed):
                self.lastQ = self.nextQ(self.lastQ, self.seed)
                self.seed += 1

        vec = self.lastQ * self.recipD

        self.lastQ = self.nextQ(self.lastQ, self.seed)
        self.seed += 1

        return vec

    def nextQ(self, lastQ, seed):
        """Generates the next value of Q in the sequence

        Keyword arguments:
        lastQ -- The last Q in the sequence
        seed -- The seed used to generate the last Q
        """

        if seed == 0:
            l = 1
        else:
            l = lowBitPos(seed)

        if l >= SobolGenerator.maxCol:
            raise RuntimeError(
                "Requested element too far into sequence. Element {0} requested but \
                element {1} is maximum".format(
                    seed + 1, SobolGenerator.atMost
                )
            )

        return numpy.bitwise_xor(lastQ, self.v[0:self.n, l - 1])


SobolGenerator.atMost = 2 ** SobolGenerator.logMax - 1
SobolGenerator.maxCol = highBitPos(SobolGenerator.atMost)
SobolGenerator.goodSeeds = numpy.array(
    [0, 0, 1, 3, 5, 8, 11, 15, 19, 23, 27, 31, 35],
    SobolGenerator.intTypeString
)
SobolGenerator.baseV = numpy.zeros(
    (SobolGenerator.nMax, SobolGenerator.logMax),
    SobolGenerator.intTypeString
)
SobolGenerator.baseV[:, 0] = 1
SobolGenerator.baseV[2:SobolGenerator.nMax, 1] = numpy.transpose([
          1, 3, 1, 3, 1, 3, 3, 1,
    3, 1, 3, 1, 3, 1, 1, 3, 1, 3,
    1, 3, 1, 3, 3, 1, 1, 1, 3, 1,
    3, 1, 3, 3, 1, 3, 1, 1, 1, 3
])
SobolGenerator.baseV[3:SobolGenerator.nMax, 2] = numpy.transpose([
             7, 5, 1, 3, 3, 7, 5,
    5, 7, 7, 1, 3, 3, 7, 5, 1, 1,
    5, 3, 7, 1, 7, 5, 1, 3, 7, 7,
    1, 1, 1, 5, 7, 7, 5, 1, 3, 3
])
SobolGenerator.baseV[5:SobolGenerator.nMax, 3] = numpy.transpose([
                        1, 7,  9,  13, 11,
    1, 3,  7,  9,  5,  13, 13, 11, 3,  15,
    5, 3,  15, 7,  9,  13, 9,  1,  11, 7,
    5, 15, 1,  15, 11, 5,  11,  1,  7,  9
])
SobolGenerator.baseV[7:SobolGenerator.nMax, 4] = numpy.transpose([
                                9,  3,  27,
    15, 29, 21, 23, 19, 11, 25, 7,  13, 17,
    1,  25, 29, 3,  31, 11, 5,  23, 27, 19,
    21, 5,  1,  17, 13, 7,  15, 9,  31, 25
])
SobolGenerator.baseV[13:SobolGenerator.nMax, 5] = numpy.transpose([
                37, 33, 7,  5,  11, 39, 63,
    59, 17, 15, 23, 29, 3,  21, 13, 31, 25,
    9,  49, 33, 19, 29, 11, 19, 27, 15, 25
])
SobolGenerator.baseV[19:SobolGenerator.nMax, 6] = numpy.transpose([
                                           13,
    33, 115, 41, 79, 17, 29,  119, 75, 73, 105,
    7,  59,  65, 21, 3,  113, 61,  89, 45, 107
])
SobolGenerator.baseV[37:SobolGenerator.nMax, 7] = numpy.transpose([
    7, 23, 39
])
SobolGenerator.baseV[0, :] = 1
SobolGenerator.poly = numpy.array(
    [
        1, 3, 7, 11, 13, 19, 25, 37, 59, 47,
        61, 55, 41, 67, 97, 91, 109, 103, 115, 131,
        193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
        213, 191, 253, 203, 211, 239, 247, 285, 369, 299
    ],
    SobolGenerator.intTypeString
)
