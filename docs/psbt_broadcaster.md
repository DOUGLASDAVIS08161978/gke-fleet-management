# PSBT Broadcaster (Mainnet-aware & Safe) - Usage & Safety Notes ✅

This document describes `scripts/psbt_broadcaster.py` — a safe CLI helper for creating PSBTs, signing with a wallet (or offline signer/HWI), finalizing, testing mempool acceptance, and broadcasting when you explicitly confirm.

Key safety rules:
- The script defaults to **dry-run** behavior and will never broadcast unless you use `--live` and explicitly pass `--broadcast` **and** a `--confirm-token` that matches the `MAINNET_BROADCAST_TOKEN` env var and you confirm interactively (type BROADCAST) or pass `--yes`.
- Broadcasting to **mainnet** requires `--network main` (or auto-detected mainnet) and the confirmation token; the script refuses to broadcast if it detects a non-main chain to avoid mistakes.
- The script performs basic address network checks (e.g., rejects `bcrt1` / `tb1` prefixes when targeting mainnet) and will call `getaddressinfo`/`validateaddress` via `bitcoin-cli` for additional validity checks.
- Always test on regtest or testnet before mainnet.
- Use PSBT + offline/hardware signing (HWI) for production mainnet usage.

Examples

1) Create a PSBT (dry-run)

  ./scripts/psbt_broadcaster.py --to <bc1...> --amount 5.0 --feerate 0.0002 --rpcwallet mywallet --psbt-out /tmp/send5btc.psbt

2) Sign with wallet locally

  ./scripts/psbt_broadcaster.py --sign --psbt /tmp/send5btc.psbt --rpcwallet mywallet

3) Sign with HWI (hardware wallet)

  ./scripts/psbt_broadcaster.py --hwi-sign --psbt /tmp/send5btc.psbt

4) Finalize and get raw hex (prints a confirmation token you must use to broadcast)

  ./scripts/psbt_broadcaster.py --finalize --psbt /tmp/psbt_signed.psbt --rpcwallet mywallet

  # The command prints a token; to broadcast on mainnet set it in env and run:
  export MAINNET_BROADCAST_TOKEN=TOKEN123
  ./scripts/psbt_broadcaster.py --network main --live --broadcast --confirm-token TOKEN123 --rawhex <rawhex>

Notes

- The script uses `bitcoin-cli` under the hood and respects `-datadir`, `-rpcuser`, `-rpcpassword` and `-rpcwallet` flags.
- The script attempts to detect the chain (`main|test|regtest`) using `getblockchaininfo`. If you plan to broadcast on mainnet, pass `--network main` to explicitly indicate intention.
- For mainnet broadcasts the script requires both the environment token and an interactive confirmation (typing `BROADCAST`) unless `--yes` is supplied.
- Review every printed PSBT / raw hex and the mempool acceptance response before broadcasting.

If you'd like, I can add a wrapper to prompt step-by-step (recommended) or integrate HWI flows more tightly (e.g., choose which device to sign with).