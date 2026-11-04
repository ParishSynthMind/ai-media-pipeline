import io

from PIL import Image

from media_pipeline.postprocess import export_asset, fit, watermark


def png(w, h, colour=(31, 111, 92)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


def test_exports_every_size_as_webp(tmp_path):
    files = export_asset(png(1200, 1200), [(1080, 1080), (1200, 628), (1080, 1920)], tmp_path, "x")
    assert [(f["width"], f["height"]) for f in files] == [(1080, 1080), (1200, 628), (1080, 1920)]
    for f in files:
        with Image.open(f["path"]) as im:
            assert im.format == "WEBP" and im.size == (f["width"], f["height"])
        assert f["bytes"] > 0


def test_png_option(tmp_path):
    files = export_asset(png(500, 500), [(100, 100)], tmp_path, "x", also_png=True)
    assert [f["format"] for f in files] == ["webp", "png"]


def test_fit_crops_without_stretching():
    img = Image.new("RGB", (200, 100), (255, 0, 0))
    for x in range(100):  # left half blue
        for y in range(100):
            img.putpixel((x, y), (0, 0, 255))
    out = fit(img, (100, 100))
    assert out.size == (100, 100)
    assert out.getpixel((10, 50))[2] > 200 and out.getpixel((90, 50))[0] > 200


def test_watermark_marks_bottom_right_only():
    img = Image.new("RGB", (800, 400), (31, 111, 92))
    out = watermark(img, "Brand Co")
    diff_corner = sum(1 for x in range(600, 800) for y in range(340, 400) if out.getpixel((x, y)) != (31, 111, 92))
    diff_top_left = sum(1 for x in range(0, 200) for y in range(0, 60) if out.getpixel((x, y)) != (31, 111, 92))
    assert diff_corner > 50 and diff_top_left == 0


def test_watermark_applied_on_export(tmp_path):
    plain = export_asset(png(400, 400), [(400, 400)], tmp_path / "a", "x")[0]
    marked = export_asset(png(400, 400), [(400, 400)], tmp_path / "b", "x", watermark_text="Brand")[0]
    with Image.open(plain["path"]) as a, Image.open(marked["path"]) as b:
        assert a.convert("RGB").getpixel((380, 390)) != b.convert("RGB").getpixel((380, 390)) or \
            list(a.convert("RGB").getdata()) != list(b.convert("RGB").getdata())
