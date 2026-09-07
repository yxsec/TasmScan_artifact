#!/usr/bin/env node
/**
 * Export instruction table from tasm TypeScript disassembler.
 *
 * Usage:
 *   node scripts/export_tasm_instructions.js <path-to-tasm-repo> [output.json]
 *
 * Example:
 *   node scripts/export_tasm_instructions.js ../tasm tasmscan/disassembler/instruction_table.json
 *
 * This generates a JSON file compatible with TasmScan's instruction_table.json format.
 * Compare with existing table to detect upstream changes.
 */

const fs = require('fs');
const path = require('path');

const tasmPath = process.argv[2];
const outputPath = process.argv[3] || '/dev/stdout';

if (!tasmPath) {
  console.error('Usage: node export_tasm_instructions.js <path-to-tasm-repo> [output.json]');
  process.exit(1);
}

const instrPath = path.join(tasmPath, 'dist/generator/instructions.js');
if (!fs.existsSync(instrPath)) {
  console.error(`Error: ${instrPath} not found. Run 'yarn build' in tasm first.`);
  process.exit(1);
}

const m = require(path.resolve(instrPath));
const entries = [];

for (const [name, spec] of Object.entries(m.instructions)) {
  // Skip fift-level macros (f* prefix) - not bytecode-level opcodes
  if (name.startsWith('f') && name.length > 1 && name[1] === name[1].toUpperCase()) {
    continue;
  }
  // Skip pseudo instructions - generated internally by decoder
  if (name.startsWith('PSEUDO_')) {
    continue;
  }

  entries.push({
    name: name,
    min: Number(spec.min),
    max: Number(spec.max),
    opcodeBits: spec.checkLen || 0,
    prefix: Number(spec.prefix || 0),
    kind: spec.kind || 'unknown',
    args: (spec.args || []).map(function convertArg(a) {
      if (!a || typeof a !== 'object') return a;
      const out = { type: a.$ || a.type || 'unknown' };
      if (a.len !== undefined) out.len = Number(a.len);
      if (a.pad !== undefined) out.pad = Number(a.pad);
      if (a.delta !== undefined) out.delta = Number(a.delta);
      if (a.arg) out.arg = convertArg(a.arg);
      if (a.refs) out.refs = convertArg(a.refs);
      if (a.bits) out.bits = convertArg(a.bits);
      return out;
    })
  });
}

const output = {
  generatedAt: new Date().toISOString(),
  source: 'tasm/dist/generator/instructions.js',
  instructions: entries
};

const json = JSON.stringify(output, (k, v) => typeof v === 'bigint' ? Number(v) : v, 2);

if (outputPath === '/dev/stdout') {
  console.log(json);
} else {
  fs.writeFileSync(outputPath, json + '\n');
  console.error(`Exported ${entries.length} instructions to ${outputPath}`);
}
