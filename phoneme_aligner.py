import argparse
from utils import (
    load_audio,
    extract_mfcc,
    load_phoneme_dict,
    align_mfcc_to_phonemes,
    print_alignment,
    plot_mfcc
)

def main():
    parser = argparse.ArgumentParser(description="Align phonemes to audio.")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (.wav or .mp3)")
    parser.add_argument("--word", type=str, required=True, help="Ukrainian word (e.g. 'привіт')")
    parser.add_argument("--phonemes", type=str, help="Comma-separated list of phonemes, overrides dictionary")

    args = parser.parse_args()

    phoneme_dict = load_phoneme_dict("data/phoneme_dict_ua.json")

    if args.phonemes:
        phonemes = args.phonemes.split(",")
    else:
        phonemes = phoneme_dict.get(args.word.lower())
        if not phonemes:
            print(f"[!] Word '{args.word}' not found in dictionary. Use --phonemes to override.")
            return

    y, sr = load_audio(args.audio)
    mfcc = extract_mfcc(y, sr)
    alignment = align_mfcc_to_phonemes(mfcc, phonemes)

    print_alignment(alignment, sr)
    plot_mfcc(mfcc, alignment, sr)


if __name__ == "__main__":
    main()
