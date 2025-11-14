if verbose:
        print(f"indices={index_display}")
        print(f"raw={','.join(map(str, raw_paths))}")
        print(f"cleaned={','.join(map(str, cleaned_paths))}")
        print(f"features={features_path}")
        print(f"models_dir={Path(args.models_dir).resolve()}")
    else:
        pass