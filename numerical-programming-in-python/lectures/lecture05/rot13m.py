import codecs

# module definitions

def rot13(input):
    """Return the rot13 encoding of an input"""
    return codecs.encode(str(input), 'rot13')


if __name__ == '__main__':
    import sys
    print(rot13(sys.argv[1]))