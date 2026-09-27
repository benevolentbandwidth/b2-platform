from tools.fake_image_detector.checks.checksums import validate_iban, validate_luhn, validate_mrz_digit


class TestValidateLuhn:
    def test_valid(self):
        assert validate_luhn("4532015112830366") is True

    def test_valid_amex(self):
        assert validate_luhn("378282246310005") is True

    def test_invalid(self):
        assert validate_luhn("4532015112830367") is False

    def test_empty(self):
        assert validate_luhn("") is False

    def test_non_digits_ignored(self):
        assert validate_luhn("4532-0151-1283-0366") is True


class TestValidateIban:
    def test_valid_de(self):
        assert validate_iban("DE89370400440532013000") is True

    def test_valid_gb(self):
        assert validate_iban("GB29NWBK60161331926819") is True

    def test_invalid_checksum(self):
        assert validate_iban("DE89370400440532013001") is False

    def test_too_short(self):
        assert validate_iban("DE89") is False

    def test_spaces_stripped(self):
        assert validate_iban("DE89 3704 0044 0532 0130 00") is True


class TestValidateMrzDigit:
    def test_valid(self):
        # "L898902C3" — document number from ICAO sample, check digit 6
        assert validate_mrz_digit("L898902C36") is True

    def test_invalid(self):
        assert validate_mrz_digit("L898902C37") is False

    def test_filler_chars_skipped(self):
        # '<' fillers treated as 0; ZE184226B< sums to 401 → check digit 1
        assert validate_mrz_digit("ZE184226B<1") is True

    def test_bad_character_returns_false(self):
        assert validate_mrz_digit("????0") is False
