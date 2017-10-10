#!/usr/bin/env python
#
#  Copyright (C) 2017 Codeplay Software Limited
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  For your convenience, a copy of the License has been included in this
#  repository.
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Codeplay's ComputeCpp SDK
#
#  extract_ir.py
#
#  Description:
#    Script to extract SPIR, SPIR-V, or other IR from a SYCL integration header.
#
#  Authors:
#    Alistair Low <alistair@codeplay.com>
#    Stephen McGroarty <stephen@codeplay.com>
#    Wojciech Nawrocki <wojciech.nawrocki@codeplay.com>
#

import os
import argparse
import sys
import re
import subprocess
import tempfile
import shutil
import struct


def parse_args():
    parser = argparse.ArgumentParser(
        description='''Finds an IR module in a SYCL integration header and
                       writes it out to a binary file. If path to llvm-dis or
                       spirv-dis is provided, tries to disassemble the binary
                       and write the disassembled text into the output file
                       instead.''')

    parser.add_argument('-i', '--input-file',
                        required=False,
                        default=sys.stdin,
                        type=argparse.FileType('r'),
                        help='The header to process, defaults to stdin.')
    parser.add_argument('-o', '--output-file',
                        required=False,
                        default='',
                        help='The file to output to, defaults to stdout.')
    parser.add_argument('-d', '--disassembler',
                        required=False,
                        default='',
                        help='''The path to llvm-dis or spirv-dis. Bytecode will
                                be disassembled only if this is provided.''')
    return parser.parse_args()


def disassemble_spir(disassembler_path, input_file_name, output_file_name):
    """
    Given two file names, uses a disassembler to disassemble the input file into
    the output file.
    """
    code = subprocess.call(
        [disassembler_path, '-o', output_file_name, input_file_name])

    if code != 0:
        raise RuntimeError('Error running disassembler! Error: ' + str(code))


def extract_ir(input_file, output_file):
    """
    Given an integration header input file and an output file open in binary
    mode, extracts the kernel IR binary from the header into the output file.
    Returns the type of IR found ('spir32', 'spir64', 'spirv32', 'spirv64') or
    '' if unknown.
    """
    start_re = re.compile(
        'unsigned char [A-Za-z-0-9_]*_bin_spirv?(64|32)\[\] = {')
    spir32_re = re.compile('spir32')
    spir64_re = re.compile('spir64')
    spirv32_re = re.compile('spirv32')
    spirv64_re = re.compile('spirv64')
    end_line = '};\n'

    reached_start = False
    bin_type = ''

    # Read the file line by line until we reach the start of the binary blob
    for line in input_file:
        if reached_start:
            if line == end_line:
                break
            else:
                line = line.replace(',', '')
                hex_list_as_string = line.split()

                for byte in hex_list_as_string:
                    output_file.write(struct.pack('B', int(byte[2:], 16)))

        if not reached_start and start_re.match(line):
            reached_start = True
            if spir32_re.search(line):
                bin_type = 'spir32'
            elif spir64_re.search(line):
                bin_type = 'spir64'
            elif spirv32_re.search(line):
                bin_type = 'spirv32'
            elif spirv64_re.search(line):
                bin_type = 'spirv64'

    for line in input_file:
        # Make sure the header has been fully read
        pass

    return bin_type


def main():
    args = parse_args()
    input_file = args.input_file
    output_file_name = args.output_file
    disassembler_path = args.disassembler

    # We have to write to the file as binary, since otherwise it can
    # add carriage returns on Windows, leading to an invalid binary.
    bin_output_file = tempfile.NamedTemporaryFile(mode='wb', delete=False)
    bin_type = extract_ir(input_file, bin_output_file)

    bin_output_file.flush()
    os.fsync(bin_output_file)
    bin_output_file.close()

    input_file.close()

    if disassembler_path != '':
        if bin_type != '':
            # Default to stdout if an output file name is not provided
            if output_file_name == '':
                output_file_name = '-'
            disassemble_spir(
                disassembler_path,
                bin_output_file.name,
                output_file_name)
        else:
            sys.stderr.write(
                'Unknown IR binary type found! Cannot disassemble.')
            exit(1)
    else:
        # disassembler argument not provided, store binary IR
        if output_file_name != '':
            # Into output file if provided
            shutil.copyfile(bin_output_file.name, output_file_name)
        else:
            # Otherwise write to stdout
            bin_output_file = open(bin_output_file.name, 'rb')
            # For version compatibility, use sys.stdout on py2 and
            # sys.stdout.buffer on py3
            out = getattr(sys.stdout, 'buffer', sys.stdout)
            out.write(bin_output_file.read())
            bin_output_file.close()

main()
