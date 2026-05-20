"""
manim_transformer/base/__init__.py
"""
from .utils import (
    # math
    softmax, scaled_dot_product, causal_mask,
    sinusoidal_pe, layer_norm, gelu, relu, silu,
    entropy, top_k_filter, top_p_filter,
    # layout
    grid_positions, auto_arrange_row, stack_positions,
    midpoint, lerp, bounding_box,
    # color
    value_to_color, heatmap_color, lerp_color,
    opacity_for_value, attention_palette,
    hex_to_rgb, rgb_to_hex,
    # text
    fmt_num, truncate_vector, latex_matrix, latex_vector,
    shape_label, wrap_text, ordinal,
    # timing
    lag_schedule, run_time_for_seq, per_item_run_time,
    # data
    make_token_sequence, make_weight_matrix,
    make_attention_weights, make_sample_logits,
    make_embedding_matrix, DEMO_VOCAB,
    # logging
    step_log, inspect_shape, validate_attention_weights,
    assert_shape, apply_dark_theme, print_quality_hint,
)
from .shapes import (
    MatrixBox, VectorBar, TokenBox, ArrowLabel,
    DimensionBrace, NumberFlow, LabeledBlock,
    MathLabel, ConnectionLine, SoftmaxCurve,
    # colors
    TOKEN_COLOR, EMBEDDING_COLOR, ATTENTION_COLOR,
    FFN_COLOR, NORM_COLOR, OUTPUT_COLOR,
    QUERY_COLOR, KEY_COLOR, VALUE_COLOR, WEIGHT_COLOR,
    HIGHLIGHT_COLOR, ARROW_COLOR, DIM_COLOR,
    POSITIVE_COLOR, NEGATIVE_COLOR, BACKGROUND_COLOR,
)
from .animations import (
    flow_through, matrix_multiply_anim, highlight_sequence,
    fade_label, pulse_glow, data_flow_arrow,
    morph_matrix, ripple_through, typewriter,
    forward_pass_pulse, attention_flow,
    dimension_change_anim, equation_reveal,
    token_stream_anim, layer_stack_anim,
    vector_addition_anim, softmax_temperature_sweep,
    zoom_into, label_appear, component_swap,
)