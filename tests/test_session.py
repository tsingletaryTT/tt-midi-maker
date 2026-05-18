from tt_midi_maker.session import MusicalContext, get_session, set_session, clear_session


def test_empty_context_is_empty():
    ctx = MusicalContext()
    assert ctx.is_empty()


def test_update_sets_field():
    ctx = MusicalContext()
    ctx2 = ctx.update(key="D minor", bpm=120)
    assert ctx2.key == "D minor"
    assert ctx2.bpm == 120


def test_update_does_not_mutate_original():
    ctx = MusicalContext(key="C major")
    ctx.update(bpm=90)
    assert ctx.bpm is None


def test_update_none_clears_field():
    ctx = MusicalContext(key="C major", bpm=120)
    ctx2 = ctx.update(key=None)
    assert ctx2.key is None
    assert ctx2.bpm == 120


def test_to_dict_omits_none():
    ctx = MusicalContext(key="D minor")
    d = ctx.to_dict()
    assert "key" in d
    assert "bpm" not in d


def test_session_store_isolation():
    set_session("A", MusicalContext(key="C major"))
    set_session("B", MusicalContext(key="D minor"))
    assert get_session("A").key == "C major"
    assert get_session("B").key == "D minor"


def test_get_unknown_session_returns_empty():
    assert get_session("nonexistent").is_empty()


def test_clear_session_removes_it():
    set_session("C", MusicalContext(bpm=120))
    clear_session("C")
    assert get_session("C").is_empty()
