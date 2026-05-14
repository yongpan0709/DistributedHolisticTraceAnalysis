from hta.configs.config import logger


def get_first_stack_on_rank(call_graph, rank):
    first_call_stack_id, first_call_stack = next(iter(call_graph.rank_to_stacks[rank].items()))
    return first_call_stack_id, first_call_stack


def get_main_stack_on_rank(call_graph, rank):
    main_stack_info = call_graph.mapping.loc[
        call_graph.mapping["rank"].eq(rank) & call_graph.mapping["label"].eq("main")
    ]

    if len(main_stack_info) != 1:
        logger.warning(f"no main stack on rank {rank}, use first stack")
        return get_first_stack_on_rank(call_graph, rank)

    main_stack_info = main_stack_info.iloc[0].to_dict()
    main_stack_id = None
    for call_stack_id in call_graph.rank_to_stacks[rank].keys():
        if (
            main_stack_info["rank"] == call_stack_id.rank
            and main_stack_info["pid"] == call_stack_id.pid
            and main_stack_info["tid"] == call_stack_id.tid
        ):
            main_stack_id = call_stack_id
            break
    assert main_stack_id is not None
    main_stack = call_graph.rank_to_stacks[rank][main_stack_id]
    return main_stack_id, main_stack
