"""
IBP Parameters
"""

def mhz_to_freq_khz(mhz):
    """
    Convert MHz to exact frequency in kHz
    """
    return {
        14: 14100,
        18: 18110,
        21: 21150,
        24: 24930,
        28: 28200
    }[mhz]

def main():
    pass

if __name__ == "__main__":
    main()
