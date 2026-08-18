# SPDX-License-Identifier: Apache-2.0
"""Collision-checked, VPN-aware fabric subnet selection."""

from __future__ import annotations

import ipaddress

import pytest

from omlx.cluster.transport import (
    LinkSetupError,
    choose_fabric_subnet,
)


def _net(cidr: str) -> ipaddress.IPv4Network:
    return ipaddress.ip_network(cidr)


def test_prefers_a_vpn_excludable_range_when_nothing_is_occupied():
    # 172.16.0.0/12 leads because full-tunnel VPNs commonly exclude it, so the
    # fabric survives a VPN that would swallow a 10.x link.
    assert str(choose_fabric_subnet([])) == "172.16.99.0/24"


def test_home_lan_does_not_push_it_off_the_preferred_range():
    assert (
        str(choose_fabric_subnet([_net("192.168.0.0/24")])) == "172.16.99.0/24"
    )


def test_skips_candidates_that_overlap_an_occupied_network():
    # A /23 at .100 covers both .100 and .101, so the next free /24 is .102.
    chosen = choose_fabric_subnet(
        [
            _net("10.0.0.0/8"),
            _net("172.16.99.0/24"),
            _net("172.16.100.0/23"),
        ]
    )
    assert str(chosen) == "172.16.102.0/24"


def test_never_reproduces_the_swallowed_10_0_1_incident():
    # 10.x is never a leading candidate; the choice stays in the VPN-excludable
    # range even when the LAN is busy.
    chosen = choose_fabric_subnet([_net("192.168.0.0/24")])
    assert chosen.overlaps(_net("172.16.0.0/12"))
    assert not chosen.overlaps(_net("10.0.1.0/24"))


def test_raises_when_every_candidate_collides():
    occupied = [_net("172.16.0.0/12"), _net("10.0.0.0/8")]
    with pytest.raises(LinkSetupError, match="free private subnet"):
        choose_fabric_subnet(occupied)
