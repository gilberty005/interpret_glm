from .datamodule import DNATrainDataModule
# from .annotation_dataset import AnnotationDNATrainDataModule
# from .fragment_annot_dataset import FragmentAnnotationDataModule
# from .gtf_dataset import GTFDataModule

# add datamodule classes here
datamodule_registry = {
    'DNATrainDataModule': DNATrainDataModule,
    # 'AnnotationDNATrainDataModule': AnnotationDNATrainDataModule,
    # 'FragmentAnnotationDataModule': FragmentAnnotationDataModule,
    # 'GTFDataModule': GTFDataModule,
}
