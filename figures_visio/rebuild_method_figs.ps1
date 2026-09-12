# Rebuild Figure 1 (e5_admm_split) and Figure 2 (e6_protocol) as native editable
# Visio shapes, faithful to make_figures_v6.py::plot_admm_split / plot_protocol.
# Coordinates are matplotlib data coords (y up): RefW = x-range, RefH = y-range.
# Page size = matplotlib figsize (inches), so pt sizes map 1:1 to the source figure.

$ErrorActionPreference = 'Stop'

. "$env:USERPROFILE\.claude\skills\visio-image-rebuilder\scripts\visio_export_formats.ps1"

# ---- palette (Okabe-Ito, matching make_figures_v6.py) ----
function RGBF([int]$r, [int]$g, [int]$b) { "RGB($r,$g,$b)" }

$script:Blue    = RGBF 0 114 178    # IDEAL #0072B2
$script:Green   = RGBF 0 158 115    # CV    #009E73
$script:Orange  = RGBF 213 94 0     # ZOH   #D55E00
$script:Ink     = RGBF 17 17 17     # INK   #111111
$script:Muted   = RGBF 58 58 58     # MUTED #3A3A3A
$script:White   = RGBF 255 255 255

# soft fills (matplotlib RGBA over white)
$script:BlueSoft   = RGBF 229 241 247   # agent_fc (blue 10%)
$script:BlueSoft2  = RGBF 235 243 248   # box_fc   (blue 8%)
$script:GreenSoft  = RGBF 224 243 238   # edge_fc  (green 12%)
$script:OrangeSoft = RGBF 251 239 230   # no_fc    (orange 10%)

# ---- coordinate mapping (matplotlib data coords -> inches; y increases up) ----
$script:RefW  = 10.0
$script:RefH  = 5.6
$script:PageW = 5.83
$script:PageH = 3.0
$script:Page  = $null

function VX([double]$x) { $script:PageW * $x / $script:RefW }
function VY([double]$y) { $script:PageH * $y / $script:RefH }

function Set-Cell($shape, [string]$cell, [string]$formula) {
    try { $shape.CellsU($cell).FormulaU = $formula } catch {}
}

function Style-Shape($shape, [string]$fill, [string]$line, [double]$linePt = 0.9, [double]$roundIn = 0.0) {
    if ($fill -eq 'none') {
        Set-Cell $shape 'FillPattern' '0'
    } else {
        Set-Cell $shape 'FillPattern' '1'
        Set-Cell $shape 'FillForegnd' $fill
    }
    if ($line -eq 'none') {
        Set-Cell $shape 'LinePattern' '0'
    } else {
        Set-Cell $shape 'LinePattern' '1'
        Set-Cell $shape 'LineColor' $line
        Set-Cell $shape 'LineWeight' "$linePt pt"
    }
    if ($roundIn -gt 0) {
        Set-Cell $shape 'Rounding' ("{0} in" -f $roundIn)
    }
}

function Set-TextStyle($shape, [string]$text, [double]$size, [string]$color, [bool]$bold, [bool]$italic, [int]$align) {
    $shape.Text = $text
    try {
        $c = $shape.Characters
        $n = $c.CharCount
        $c.Begin = 0
        $c.End = $n
        $c.CharProps(0) = $script:FontId
    } catch {}
    Set-Cell $shape 'Char.AsianFont' ([string]$script:FontId)
    Set-Cell $shape 'Char.ComplexScriptFont' ([string]$script:FontId)
    Set-Cell $shape 'Char.Size' "$size pt"
    Set-Cell $shape 'Char.Color' $color
    $style = 0
    if ($bold) { $style += 1 }
    if ($italic) { $style += 2 }
    Set-Cell $shape 'Char.Style' ([string]$style)
    Set-Cell $shape 'Para.HorzAlign' ([string]$align)
    Set-Cell $shape 'VerticalAlign' '1'
    foreach ($m in 'LeftMargin','RightMargin','TopMargin','BottomMargin') {
        Set-Cell $shape $m '2 pt'
    }
}

function Rect([double]$x, [double]$y, [double]$w, [double]$h, [string]$fill, [string]$line, [double]$linePt = 0.9, [double]$roundIn = 0.0) {
    $s = $script:Page.DrawRectangle((VX $x), (VY ($y + $h)), (VX ($x + $w)), (VY $y))
    Style-Shape $s $fill $line $linePt $roundIn
    return $s
}

function Text([double]$x, [double]$y, [double]$w, [double]$h, [string]$txt, [double]$size = 8, [string]$color = '', [bool]$bold = $false, [bool]$italic = $false, [int]$align = 1, [double]$angle = 0.0) {
    if ($color -eq '') { $color = $script:Ink }
    $s = $script:Page.DrawRectangle((VX $x), (VY ($y + $h)), (VX ($x + $w)), (VY $y))
    Style-Shape $s 'none' 'none'
    Set-TextStyle $s $txt $size $color $bold $italic $align
    if ($angle -ne 0.0) { Set-Cell $s 'Angle' ("{0} deg" -f $angle) }
    return $s
}

function Line([double]$x1, [double]$y1, [double]$x2, [double]$y2, [string]$color, [double]$linePt = 1.0, [bool]$arrowEnd = $false, [bool]$arrowBegin = $false) {
    $s = $script:Page.DrawLine((VX $x1), (VY $y1), (VX $x2), (VY $y2))
    Set-Cell $s 'LineColor' $color
    Set-Cell $s 'LineWeight' "$linePt pt"
    Set-Cell $s 'LinePattern' '1'
    if ($arrowEnd) { Set-Cell $s 'EndArrow' '4' }
    if ($arrowBegin) { Set-Cell $s 'BeginArrow' '4' }
    return $s
}

function Diamond([double]$cx, [double]$cy, [double]$hw, [double]$hh) {
    $c = $script:Ink
    Line ($cx - $hw) $cy $cx ($cy + $hh) $c 1.0 | Out-Null
    Line $cx ($cy + $hh) ($cx + $hw) $cy $c 1.0 | Out-Null
    Line ($cx + $hw) $cy $cx ($cy - $hh) $c 1.0 | Out-Null
    Line $cx ($cy - $hh) ($cx - $hw) $cy $c 1.0 | Out-Null
}

function Fix-SvgFont([string]$path) {
    # Visio's SVG export on a Chinese-locale Windows falls back to SimSun (宋体)
    # for Greek/math glyphs even though the .vsdx text is all Times New Roman.
    # Rewrite the exported SVG so every font-family is Times New Roman (TNR has all glyphs).
    if (-not (Test-Path -LiteralPath $path)) { return }
    $song = [string][char]0x5B8B + [char]0x4F53   # 宋体
    $txt = [IO.File]::ReadAllText($path, [Text.Encoding]::UTF8)
    if ($txt.Contains($song)) {
        $txt = $txt.Replace($song, 'Times New Roman')
        [IO.File]::WriteAllText($path, $txt, (New-Object Text.UTF8Encoding($false)))
    }
}

# ============================================================================
# Figure 1 — edge-splitting ADMM schematic
# ============================================================================
function Draw-Fig1 {
    $B = $script:Blue; $G = $script:Green; $O = $script:Orange
    $I = $script:Ink; $M = $script:Muted; $W = $script:White

    # agent boxes
    Rect 0.4 2.275 2.6 2.15 $script:BlueSoft $B 1.2 0.05 | Out-Null
    Rect 7.0 2.275 2.6 2.15 $script:BlueSoft $B 1.2 0.05 | Out-Null
    # edge box
    Rect 3.5 2.275 3.0 2.15 $script:GreenSoft $G 1.2 0.05 | Out-Null

    # output sub-boxes (agent) — orange border
    Rect 0.65 2.35 2.1 0.62 $W $O 0.9 0.03 | Out-Null
    Rect 7.25 2.35 2.1 0.62 $W $O 0.9 0.03 | Out-Null
    # copy sub-boxes (edge) — green border
    Rect 3.73 3.20 1.1 0.62 $W $G 0.9 0.03 | Out-Null
    Rect 5.17 3.20 1.1 0.62 $W $G 0.9 0.03 | Out-Null

    # agent text
    Text 0.70 3.86 2.00 0.36 'Agent i' 8.5 $I $true | Out-Null
    Text 7.30 3.86 2.00 0.36 'Agent j' 8.5 $I $true | Out-Null
    Text 0.70 3.40 2.00 0.30 'min f_i(x_i, u_i)' 7.5 $I | Out-Null
    Text 7.30 3.40 2.00 0.30 'min f_j(x_j, u_j)' 7.5 $I | Out-Null
    Text 0.60 3.02 2.20 0.28 's.t. dynamics, |u_i| ≤ ū' 6.8 $M | Out-Null
    Text 7.20 3.02 2.20 0.28 's.t. dynamics, |u_j| ≤ ū' 6.8 $M | Out-Null
    Text 0.65 2.40 2.10 0.50 's_{e,i} = C·x_i − d_i' 7.2 $I | Out-Null
    Text 7.25 2.40 2.10 0.50 's_{e,j} = C·x_j − d_j' 7.2 $I | Out-Null

    # edge text
    Text 4.00 3.86 2.00 0.36 'Edge e = (i, j)' 8.5 $I $true | Out-Null
    Text 3.73 3.32 1.10 0.44 'z_{e,i}' 7.6 $I | Out-Null
    Text 5.17 3.32 1.10 0.44 'z_{e,j}' 7.6 $I | Out-Null
    Text 2.80 2.24 4.40 0.32 'couple γ·w_e·‖z_{e,i}−z_{e,j}‖_W²' 6.5 $G | Out-Null

    # coupling spring (double-headed) between copies
    Line 4.83 3.51 5.17 3.51 $G 1.2 $true $true | Out-Null

    # consensus arrows + labels
    Line 3.0 2.66 4.31 3.19 $O 1.2 $true | Out-Null
    Line 7.0 2.66 5.69 3.19 $O 1.2 $true | Out-Null
    Text 3.22 3.52 0.80 0.26 'consensus' 6.3 $M | Out-Null
    Text 5.98 3.52 0.80 0.26 'consensus' 6.3 $M | Out-Null
    Text 3.12 2.16 1.00 0.26 'η_{e,i}' 7.0 $O | Out-Null
    Text 5.88 2.16 1.00 0.26 'η_{e,j}' 7.0 $O | Out-Null

    # bottom annotation
    Text 0.05 0.05 9.90 0.60 'each endpoint output is duplicated at the edge; the scaled dual η_e enforces s_{e,·} = z_{e,·}, and only z_e is coupled' 6.5 $M | Out-Null
}

# ============================================================================
# Figure 2 — confidence-decay denial protocol flowchart
# ============================================================================
function Draw-Fig2 {
    $B = $script:Blue; $G = $script:Green; $O = $script:Orange
    $I = $script:Ink; $M = $script:Muted; $W = $script:White

    # start
    Rect 3.2 10.025 3.6 0.95 $script:BlueSoft2 $B 1.2 0.05 | Out-Null
    Text 3.2 10.06 3.6 0.88 "MPC step k:`ntry to receive neighbor messages" 7.2 $I | Out-Null
    # decision diamond
    Diamond 5.0 8.8 1.15 0.675
    Text 4.2 8.44 1.6 0.72 "communication`nactive?" 7.0 $I | Out-Null
    Line 5.0 10.02 5.0 9.48 $I 1.0 $true | Out-Null

    # yes branch (left)
    Rect 0.95 7.125 2.7 0.95 $script:BlueSoft2 $B 1.2 0.05 | Out-Null
    Text 0.95 7.16 2.7 0.88 "exact edge update`n(eq. 5)" 7.2 $I | Out-Null
    Line 4.35 8.35 2.6 8.05 $G 1.2 $true | Out-Null
    Text 2.90 8.42 0.60 0.28 'yes' 6.5 $G $false $true | Out-Null

    # no branch (right)
    Rect 6.1 7.125 3.2 0.95 $script:OrangeSoft $O 1.2 0.05 | Out-Null
    Text 6.1 7.16 3.2 0.88 "predict remote output`nŝ (ZOH / CV)" 7.2 $I | Out-Null
    Line 5.65 8.35 7.4 8.05 $O 1.2 $true | Out-Null
    Text 6.30 8.42 0.60 0.28 'no' 6.5 $O $false $true | Out-Null

    Rect 6.1 5.725 3.2 0.95 $script:OrangeSoft $O 1.2 0.05 | Out-Null
    Text 6.1 5.76 3.2 0.88 "compute age τ of`nlast message" 7.2 $I | Out-Null
    Line 7.7 7.12 7.7 6.68 $O 1.1 $true | Out-Null

    Rect 6.1 4.325 3.2 0.95 $script:OrangeSoft $O 1.2 0.05 | Out-Null
    Text 6.1 4.36 3.2 0.88 "decay weight`nw · e^(−τ/T_w)" 7.2 $I | Out-Null
    Line 7.7 5.72 7.7 5.28 $O 1.1 $true | Out-Null

    Rect 6.1 2.925 3.2 0.95 $script:OrangeSoft $O 1.2 0.05 | Out-Null
    Text 6.1 2.96 3.2 0.88 "inexact (half) edge`nupdate" 7.2 $I | Out-Null
    Line 7.7 4.32 7.7 3.88 $O 1.1 $true | Out-Null

    # merge
    Rect 2.8 1.375 4.4 1.05 $W $I 1.2 0.05 | Out-Null
    Text 2.8 1.42 4.4 0.96 "solve local QP + dual update`napply first control u_0, shift horizon" 7.2 $I | Out-Null
    Line 2.3 7.12 3.6 2.42 $G 1.1 $true | Out-Null
    Line 7.7 2.92 6.6 2.42 $O 1.1 $true | Out-Null

    # loop back
    Line 2.8 1.9 0.4 1.9 $I 1.0 | Out-Null
    Line 0.4 1.9 0.4 10.5 $I 1.0 | Out-Null
    Line 0.4 10.5 3.2 10.5 $I 1.0 $true | Out-Null
    Text 0.00 5.90 0.60 0.60 'repeat' 6.3 $M $false $false 1 90 | Out-Null
}

# ============================================================================
# Main
# ============================================================================
$outDir = $PSScriptRoot
if (-not (Test-Path -LiteralPath $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }

$visio = $null
try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $true

    # --- Figure 1 ---
    $script:RefW = 10.0; $script:RefH = 5.6
    $script:PageW = 5.83; $script:PageH = 3.0
    $doc1 = $visio.Documents.Add("")
    $script:Page = $doc1.Pages.Item(1)
    $script:FontId = $doc1.Fonts.Item('Times New Roman').ID
    $script:Page.PageSheet.CellsU('PageWidth').FormulaU = '5.83 in'
    $script:Page.PageSheet.CellsU('PageHeight').FormulaU = '3 in'
    Draw-Fig1
    $fig1vsdx = Join-Path $outDir 'e5_admm_split.vsdx'
    $doc1.SaveAs($fig1vsdx)
    Write-Output "Saved: $fig1vsdx"
    Export-VisioPageFormats -Document $doc1 -Page $script:Page -SourcePath $fig1vsdx `
        -Formats @('svg','pdf','png') -OutputDir $outDir -OutputBaseName 'e5_admm_split'
    Fix-SvgFont (Join-Path $outDir 'e5_admm_split.svg')
    $doc1.Close()

    # --- Figure 2 ---
    $script:RefW = 10.0; $script:RefH = 11.4
    $script:PageW = 5.83; $script:PageH = 3.6
    $doc2 = $visio.Documents.Add("")
    $script:Page = $doc2.Pages.Item(1)
    $script:FontId = $doc2.Fonts.Item('Times New Roman').ID
    $script:Page.PageSheet.CellsU('PageWidth').FormulaU = '5.83 in'
    $script:Page.PageSheet.CellsU('PageHeight').FormulaU = '3.6 in'
    Draw-Fig2
    $fig2vsdx = Join-Path $outDir 'e6_protocol.vsdx'
    $doc2.SaveAs($fig2vsdx)
    Write-Output "Saved: $fig2vsdx"
    Export-VisioPageFormats -Document $doc2 -Page $script:Page -SourcePath $fig2vsdx `
        -Formats @('svg','pdf','png') -OutputDir $outDir -OutputBaseName 'e6_protocol'
    Fix-SvgFont (Join-Path $outDir 'e6_protocol.svg')
    $doc2.Close()

    Write-Output "DONE"
} finally {
    if ($visio -ne $null) {
        try { $visio.Quit() } catch {}
        try { [System.Runtime.Interopservices.Marshal]::ReleaseComObject($visio) | Out-Null } catch {}
    }
}
