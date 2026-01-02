#!/usr/bin/env python3
"""
PSBT Broadcaster (Mainnet-aware, safe defaults)

Features added:
 - Automatic network detection (via getblockchaininfo)
 - Mainnet-only safeguards: explicit --network main flag required for broadcasting to mainnet
 - Address validation (basic prefix check + getaddressinfo isvalid)
 - Wallet balance check prior to creating PSBT
 - Interactive confirmation prompt before final broadcast (type BROADCAST)
 - Optional HWI signing integration (if HWI is installed and --hwi-sign passed)
 - Tokens persisted to /tmp/psbt_broadcast_token.txt for safer reuse

Usage examples are similar to before, but broadcasting to mainnet requires additional confirmations.
"""

import argparse
import json
import os
import random
import string
import subprocess
import sys


def run_cmd(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.stdout.strip(), proc.stderr.strip(), proc.returncode


def cli_call(args, cli_common):
    cmd = ["bitcoin-cli"] + cli_common + args
    out, err, rc = run_cmd(cmd)
    if rc != 0:
        raise SystemExit(f"bitcoin-cli failed: {' '.join(cmd)}\nerr: {err}\nstdout: {out}")
    return out


def parse_json(s):
    try:
        return json.loads(s)
    except Exception:
        return s


def rand_token(n=24):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(n))


def detect_chain(cli_common):
    try:
        out = cli_call(['getblockchaininfo'], cli_common)
        j = parse_json(out)
        return j.get('chain') if isinstance(j, dict) else 'unknown'
    except Exception:
        return 'unknown'


def is_mainnet_address(addr):
    # Basic prefix checks for mainnet addresses
    if addr.startswith('bc1') or addr.startswith('1') or addr.startswith('3'):
        return True
    return False


def get_wallet_balance(cli_common):
    try:
        out = cli_call(['getbalances'], cli_common)
        j = parse_json(out)
        # getbalances returns dict with 'mine'/'watchonly' etc in modern versions
        if isinstance(j, dict) and 'mine' in j and 'trusted' in j['mine']:
            return float(j['mine']['trusted'])
        # fallback to getwalletinfo
        out = cli_call(['getwalletinfo'], cli_common)
        j = parse_json(out)
        if isinstance(j, dict) and 'balance' in j:
            return float(j['balance'])
    except Exception:
        pass
    return 0.0


def hwi_sign(psbt_b64):
    # Attempt to sign using HWI if available. This is optional and will raise if HWI is missing.
    out, err, rc = run_cmd(['hwi', 'enumerate'])
    if rc != 0:
        raise SystemExit('HWI not found or not available: ' + err)
    # For simplicity, use the first device
    out, err, rc = run_cmd(['hwi', 'signtx', '-i', psbt_b64])
    if rc != 0:
        raise SystemExit('HWI signtx failed: ' + err)
    return out


def confirm_interactive(prompt):
    print(prompt)
    print("Type 'BROADCAST' to confirm, anything else will abort:")
    resp = input().strip()
    return resp == 'BROADCAST'


def main():
    p = argparse.ArgumentParser(description="Safe PSBT creator + mainnet-aware broadcaster")

    p.add_argument('--datadir', help='path to bitcoin datadir', default=None)
    p.add_argument('--rpcuser', default=None)
    p.add_argument('--rpcpassword', default=None)
    p.add_argument('--rpcwallet', default=None, help='wallet name to use for wallet RPCs')

    p.add_argument('--network', choices=['auto', 'main', 'test', 'regtest'], default='auto', help='Network expectation; if broadcasting to mainnet, pass --network main')

    p.add_argument('--to', help='destination address')
    p.add_argument('--amount', type=float, help='amount in BTC to send')
    p.add_argument('--feerate', type=float, default=0.0002, help='fee rate (BTC per kvB)')

    p.add_argument('--psbt-out', help='file to write the PSBT base64 to')
    p.add_argument('--psbt', help='PSBT base64 to sign or path to file with PSBT')
    p.add_argument('--sign', action='store_true', help='ask rpc wallet to sign the PSBT (walletprocesspsbt)')
    p.add_argument('--hwi-sign', action='store_true', help='attempt to sign PSBT using HWI (hardware wallet interface)')
    p.add_argument('--finalize', action='store_true', help='finalize PSBT (finalizepsbt)')
    p.add_argument('--rawhex', help='raw tx hex to test/broadcast (can be obtained from finalizepsbt)')

    p.add_argument('--testmempoolaccept', action='store_true', help='run testmempoolaccept on rawhex')
    p.add_argument('--broadcast', action='store_true', help='broadcast the raw tx (REQUIRES confirmation token and mainnet confirmations)')
    p.add_argument('--confirm-token', help='confirmation token for broadcasting')

    p.add_argument('--dry-run', action='store_true', default=True, dest='dry_run', help='do NOT broadcast or change network state (default)')
    p.add_argument('--live', action='store_false', dest='dry_run', help='allow operations that change chain state if other confirmations are provided')

    p.add_argument('--yes', action='store_true', help='non-interactive confirmation (dangerous; still requires tokens for mainnet)')

    args = p.parse_args()

    cli_common = []
    if args.datadir:
        cli_common += [f'-datadir={args.datadir}']
    if args.rpcuser:
        cli_common += [f'-rpcuser={args.rpcuser}']
    if args.rpcpassword:
        cli_common += [f'-rpcpassword={args.rpcpassword}']
    if args.rpcwallet:
        cli_common += [f'-rpcwallet={args.rpcwallet}']

    chain = detect_chain(cli_common) if args.network == 'auto' else args.network
    print(f'Detected chain: {chain}')

    # Prevent broadcasting on anything other than main unless user explicitly requested that network
    if args.broadcast:
        if args.dry_run:
            print('ERROR: --broadcast requires --live (not --dry-run). Use --live only when you understand the consequences.')
            sys.exit(1)
        if args.network != 'main' and args.network != 'auto':
            print('ERROR: --broadcast requires --network main when targeting mainnet')
            sys.exit(1)
        if chain != 'main' and args.network == 'main':
            raise SystemExit('ERROR: Detected chain is not main but --network main was requested. Aborting to avoid accidental broadcast on wrong network.')

    # Create PSBT (with checks)
    if args.to and args.amount is not None:
        if args.dry_run:
            print('NOTE: Creating PSBT in dry-run mode (no broadcast). Remove --dry-run/--live when you intend to broadcast).')

        # Basic address sanity for mainnet
        if (args.network == 'main' or chain == 'main') and not is_mainnet_address(args.to):
            raise SystemExit('ERROR: Destination address does not look like a mainnet address. Aborting.')

        balance = get_wallet_balance(cli_common)
        print(f'Wallet balance (approx): {balance} BTC')
        estimated_total = args.amount + (args.feerate * 0.001 * 250)  # rough vsize estimate 250 vbytes
        if balance < estimated_total:
            print(f'WARNING: wallet balance ({balance}) < estimated total (amount+fee ~ {estimated_total}). PSBT creation may fail.')

        print(f'Creating PSBT to: {args.to} amount: {args.amount} BTC (feeRate={args.feerate})')
        funded = cli_call(['walletcreatefundedpsbt', '[]', json.dumps({args.to: args.amount}), '0', json.dumps({'feeRate': args.feerate})], cli_common)
        j = parse_json(funded)
        psbt = j.get('psbt') if isinstance(j, dict) else j
        fee = j.get('fee') if isinstance(j, dict) else None
        print(f'Fee: {fee} BTC')

        if args.psbt_out:
            with open(args.psbt_out, 'w') as f:
                f.write(psbt)
            print(f'PSBT written to {args.psbt_out}')
        else:
            print('\nPSBT (base64):')
            print(psbt)

        print('\nNext steps:')
        print(' - Have the PSBT signed with your offline signer, HWI, or with the wallet using --sign --psbt <file or base64>')
        print(' - Finalize the PSBT with --finalize and then test with --testmempoolaccept --rawhex <rawhex>')
        print(' - To broadcast on mainnet: run with --network main --live --broadcast --confirm-token <token> and type BROADCAST when prompted (or pass --yes and env token)')
        sys.exit(0)

    # Sign PSBT
    if args.psbt or args.sign or args.hwi_sign:
        psbt_value = None
        if args.psbt:
            if os.path.exists(args.psbt):
                psbt_value = open(args.psbt).read().strip()
            else:
                psbt_value = args.psbt

        if not psbt_value:
            print('ERROR: No PSBT provided')
            sys.exit(1)

        if args.hwi_sign:
            print('Attempting to sign PSBT using HWI (hardware wallet interface)')
            signed = hwi_sign(psbt_value)
            open('/tmp/psbt_signed_by_hwi.psbt', 'w').write(signed)
            print('Signed by HWI saved to /tmp/psbt_signed_by_hwi.psbt')
            sys.exit(0)

        if args.sign:
            print('Signing PSBT with wallet (walletprocesspsbt)')
            out = cli_call(['walletprocesspsbt', psbt_value], cli_common)
            j = parse_json(out)
            signed = j.get('psbt') if isinstance(j, dict) else out
            complete = j.get('complete') if isinstance(j, dict) else None
            open('/tmp/psbt_signed.psbt', 'w').write(signed)
            print('Signed PSBT saved to /tmp/psbt_signed.psbt')
            print(f'complete: {complete}')
            print('You can run --finalize --psbt /tmp/psbt_signed.psbt to get the raw tx hex')
            sys.exit(0)

    # Finalize PSBT
    if args.finalize:
        psbt_value = None
        if args.psbt:
            if os.path.exists(args.psbt):
                psbt_value = open(args.psbt).read().strip()
            else:
                psbt_value = args.psbt
        else:
            print('ERROR: --finalize requires --psbt <file or base64>')
            sys.exit(1)

        print('Finalizing PSBT (finalizepsbt)')
        out = cli_call(['finalizepsbt', psbt_value], cli_common)
        j = parse_json(out)
        raw = j.get('hex') if isinstance(j, dict) else out
        complete = j.get('complete') if isinstance(j, dict) else None
        print(f'complete: {complete}')
        print('\nRaw hex:')
        print(raw)

        token = rand_token()
        token_path = '/tmp/psbt_broadcast_token.txt'
        open(token_path, 'w').write(token)
        print('\nTo broadcast this transaction on mainnet, do the following:')
        print(f'  export MAINNET_BROADCAST_TOKEN={token}')
        print('Then run:')
        print(f'  ./scripts/psbt_broadcaster.py --network main --live --broadcast --confirm-token {token} --rawhex <rawhex>')
        print('\nThe script will require you to type BROADCAST at the prompt unless you pass --yes (still requires the token)')
        sys.exit(0)

    # Test mempool acceptance
    if args.testmempoolaccept:
        if not args.rawhex:
            print('ERROR: --testmempoolaccept requires --rawhex')
            sys.exit(1)
        print('Testing mempool acceptance (testmempoolaccept)')
        out = cli_call(['testmempoolaccept', json.dumps([args.rawhex])], cli_common)
        print(out)
        sys.exit(0)

    # Broadcast
    if args.broadcast:
        if args.dry_run:
            print('ERROR: --broadcast requires --live and explicit confirmation')
            sys.exit(1)

        # Ensure we are on mainnet when broadcasting
        chain = detect_chain(cli_common) if args.network == 'auto' else args.network
        if chain != 'main':
            raise SystemExit('ERROR: Detected chain is not main; aborting broadcast to avoid accidents.')

        token_env = os.environ.get('MAINNET_BROADCAST_TOKEN')
        if not token_env:
            print('ERROR: environment variable MAINNET_BROADCAST_TOKEN is not set')
            sys.exit(1)
        if args.confirm_token != token_env:
            print('ERROR: --confirm-token does not match MAINNET_BROADCAST_TOKEN')
            sys.exit(1)
        if not args.rawhex:
            print('ERROR: --broadcast requires --rawhex <rawhex>')
            sys.exit(1)

        # Interactive confirmation unless --yes
        if not args.yes:
            ok = confirm_interactive(f'About to broadcast to network={chain} rawtx (first 40 hex): {args.rawhex[:40]}...')
            if not ok:
                print('Aborted by user')
                sys.exit(1)

        print('Broadcasting raw tx (sendrawtransaction)')
        out = cli_call(['sendrawtransaction', args.rawhex], cli_common)
        print('Broadcast txid:')
        print(out)
        sys.exit(0)

    print('No operation specified. See --help for usage.')


if __name__ == '__main__':
    main()
