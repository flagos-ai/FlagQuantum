"""Express all 67 signed clauses as one native phase-oracle intent."""

from math import pi

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, allocate, control, phase, qfunc


@qfunc
def main(variables: Output[QArray[QBit]]) -> None:
    allocate(67, variables)
    control(
        (variables[0] or not variables[29] or variables[48])
        and (variables[1] or not variables[31] or variables[66])
        and (variables[2] or not variables[3] or variables[6])
        and (variables[3] or not variables[16] or variables[56])
        and (variables[4] or not variables[47] or variables[65])
        and (variables[5] or not variables[24] or variables[47])
        and (variables[6] or not variables[14] or variables[2])
        and (variables[7] or not variables[55] or variables[5])
        and (variables[8] or not variables[56] or variables[37])
        and (variables[9] or not variables[60] or variables[31])
        and (variables[10] or not variables[34] or variables[51])
        and (variables[11] or not variables[13] or variables[58])
        and (variables[12] or not variables[64] or variables[0])
        and (variables[13] or not variables[57] or variables[25])
        and (variables[14] or not variables[40] or variables[33])
        and (variables[15] or not variables[17] or variables[50])
        and (variables[16] or not variables[33] or variables[54])
        and (variables[17] or not variables[53] or variables[7])
        and (variables[18] or not variables[26] or variables[28])
        and (variables[19] or not variables[39] or variables[52])
        and (variables[20] or not variables[1] or variables[40])
        and (variables[21] or not variables[65] or variables[30])
        and (variables[22] or not variables[21] or variables[36])
        and (variables[23] or not variables[28] or variables[46])
        and (variables[24] or not variables[62] or variables[32])
        and (variables[25] or not variables[19] or variables[59])
        and (variables[26] or not variables[61] or variables[45])
        and (variables[27] or not variables[42] or variables[20])
        and (variables[28] or not variables[8] or variables[64])
        and (variables[29] or not variables[22] or variables[13])
        and (variables[30] or not variables[27] or variables[34])
        and (variables[31] or not variables[38] or variables[15])
        and (variables[32] or not variables[52] or variables[22])
        and (variables[33] or not variables[7] or variables[18])
        and (variables[34] or not variables[4] or variables[11])
        and (variables[35] or not variables[43] or variables[16])
        and (variables[36] or not variables[58] or variables[4])
        and (variables[37] or not variables[11] or variables[1])
        and (variables[38] or not variables[66] or variables[10])
        and (variables[39] or not variables[63] or variables[43])
        and (variables[40] or not variables[48] or variables[62])
        and (variables[41] or not variables[10] or variables[8])
        and (variables[42] or not variables[35] or variables[14])
        and (variables[43] or not variables[50] or variables[19])
        and (variables[44] or not variables[51] or variables[55])
        and (variables[45] or not variables[2] or variables[24])
        and (variables[46] or not variables[59] or variables[29])
        and (variables[47] or not variables[37] or variables[21])
        and (variables[48] or not variables[41] or variables[44])
        and (variables[49] or not variables[6] or variables[17])
        and (variables[50] or not variables[20] or variables[42])
        and (variables[51] or not variables[0] or variables[49])
        and (variables[52] or not variables[54] or variables[23])
        and (variables[53] or not variables[46] or variables[9])
        and (variables[54] or not variables[23] or variables[26])
        and (variables[55] or not variables[18] or variables[39])
        and (variables[56] or not variables[44] or variables[41])
        and (variables[57] or not variables[45] or variables[61])
        and (variables[58] or not variables[25] or variables[12])
        and (variables[59] or not variables[49] or variables[63])
        and (variables[60] or not variables[15] or variables[3])
        and (variables[61] or not variables[32] or variables[27])
        and (variables[62] or not variables[12] or variables[53])
        and (variables[63] or not variables[30] or variables[60])
        and (variables[64] or not variables[9] or variables[38])
        and (variables[65] or not variables[36] or variables[35])
        and (variables[66] or not variables[5] or variables[57]),
        lambda: phase(1, pi),
    )


def build_program() -> synth.Program:
    return synth.synthesize(
        main, max_qubits=150, max_workspace=83, memory_limit_bytes=4800
    )


if __name__ == "__main__":
    program = build_program()
    fq.run(program)
    print(program.report["resources"])
    print(program.report["execution_profile"])
