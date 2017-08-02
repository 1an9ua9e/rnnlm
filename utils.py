# -*- coding: utf-8 -*-


def tomap(args):
    return getattr(args[0],args[1])(*args[2:])

